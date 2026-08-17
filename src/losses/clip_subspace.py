from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# ── Encoder loaders ───────────────────────────────────────────────────────────

def load_clip_encoder(device):
    """Load frozen CLIP ViT-B/32 encoder and its preprocessor."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, "clip"


def load_dino_encoder(device, model_name="dinov2_vitb14"):
    """Load frozen DINOv2 encoder with standard ImageNet preprocessing."""
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, preprocess, "dino"


# ── Internal feature extraction ───────────────────────────────────────────────

@torch.no_grad()
def _extract_features(model, preprocess, raw_data, indices, device, encoder_type="clip", batch_size=256):
    """Extract normalized features from raw numpy images at given dataset indices (CIFAR-style)."""
    features = []
    idx_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
    for start in range(0, len(idx_list), batch_size):
        batch_idx = idx_list[start : start + batch_size]
        imgs = torch.stack([preprocess(Image.fromarray(raw_data[i])) for i in batch_idx]).to(device)
        if encoder_type == "clip":
            feats = model.encode_image(imgs).float()
        else:
            feats = model(imgs).float()
        features.append(F.normalize(feats, dim=-1))
    return torch.cat(features, dim=0)  # (N, D)


@torch.no_grad()
def _extract_features_from_loader(model, loader, device, encoder_type="dino"):
    """Extract normalized features from a DataLoader (DomainNet / file-based datasets)."""
    features = []
    for xb, _ in loader:
        xb = xb.to(device)
        if encoder_type == "clip":
            feats = model.encode_image(xb).float()
        else:
            feats = model(xb).float()
        features.append(F.normalize(feats, dim=-1))
    return torch.cat(features, dim=0)  # (N, D)


def subspace_weights_from_features(
    forget_feat: torch.Tensor,
    retain_feat: torch.Tensor,
    n_components: int = 10,
):
    """
    Per-sample subspace weights from already-extracted encoder features.

    Factored out of compute_domainnet_subspace_weights so that a hyperparameter
    sweep can extract features ONCE and then vary n_components (and lambda_clip,
    which does not affect the weights at all) without paying for DINOv2 again.
    The arithmetic is identical to the original inline version.

    w_i = fraction of retain sample i's globally-centered feature energy that
    lies inside the top-k PCA subspace of the forget set.
    """
    global_mean = torch.cat([forget_feat, retain_feat], dim=0).mean(dim=0)

    forget_centered = forget_feat - global_mean.unsqueeze(0)
    _, _, Vt = torch.linalg.svd(forget_centered, full_matrices=False)
    subspace = Vt[:n_components]  # (k, D)

    retain_centered = retain_feat - global_mean.unsqueeze(0)
    proj_energy  = (retain_centered @ subspace.T).pow(2).sum(dim=1)
    total_energy = retain_centered.pow(2).sum(dim=1).clamp(min=1e-8)
    weights      = (proj_energy / total_energy).clamp(min=0, max=1)

    return weights, subspace, global_mean


def style_subspace_from_features(
    feats: torch.Tensor,
    class_labels: torch.Tensor,
    domain_labels: torch.Tensor,
    n_components: int = 10,
    min_per_cell: int = 5,
    exclude_class: int | None = None,
):
    """
    Estimate the rendering-STYLE subspace from cross-domain variation.

    For every (class, domain) cell we take the centroid and subtract that class's
    overall centroid. The residual is "what changes when this same concept is
    re-rendered in a different style". Pooling those residuals over many classes
    and taking the top-m principal directions gives a subspace that spans style
    while being largely free of any single concept's identity.

    `exclude_class` drops the forget class from the estimate, so the style
    subspace is built with NO supervision about the concept being erased -- which
    matches deployment, where you do not have labelled harmful examples in every
    rendering style.

    Returns an orthonormal (m, D) basis.
    """
    diffs = []
    for c in torch.unique(class_labels):
        if exclude_class is not None and int(c) == int(exclude_class):
            continue
        m_c = class_labels == c
        if int(m_c.sum()) < min_per_cell:
            continue
        mu_c = feats[m_c].mean(dim=0)
        for d in torch.unique(domain_labels):
            m = m_c & (domain_labels == d)
            if int(m.sum()) < min_per_cell:
                continue
            diffs.append(feats[m].mean(dim=0) - mu_c)

    if len(diffs) < 2:
        raise ValueError(
            f"style subspace needs >=2 (class, domain) cells with >={min_per_cell} samples; got {len(diffs)}"
        )
    D = torch.stack(diffs, dim=0)                      # (n_cells, D)
    _, _, Vt = torch.linalg.svd(D, full_matrices=False)
    return Vt[: min(n_components, Vt.shape[0])]        # orthonormal rows


def style_invariant_weights_from_features(
    forget_feat: torch.Tensor,
    retain_feat: torch.Tensor,
    style_subspace: torch.Tensor,
    n_components: int = 5,
):
    """
    Per-sample weights computed AFTER projecting out the style subspace.

    The stock weighting (subspace_weights_from_features) builds its subspace from
    the forget cell, which mixes concept identity with rendering style: a retain
    sample scores high either because it depicts something similar OR merely
    because it is drawn in the same style. That conflation is exactly what ties
    style-invariance to collateral damage.

    Here we deflate the style directions first, so a retain sample scores high
    only if it shares the forget CONCEPT -- which should raise transfer to the
    same concept in other domains while sparing style-confounded neighbours.

    Returns (weights (N_retain,), concept_subspace (k, D)).
    """
    S = style_subspace
    global_mean = torch.cat([forget_feat, retain_feat], dim=0).mean(dim=0)

    def deflate(x: torch.Tensor) -> torch.Tensor:
        xc = x - global_mean.unsqueeze(0)
        return xc - (xc @ S.T) @ S     # S has orthonormal rows

    fz = deflate(forget_feat)
    rz = deflate(retain_feat)

    _, _, Vt = torch.linalg.svd(fz, full_matrices=False)
    concept_subspace = Vt[:n_components]

    proj_energy  = (rz @ concept_subspace.T).pow(2).sum(dim=1)
    total_energy = rz.pow(2).sum(dim=1).clamp(min=1e-8)
    weights      = (proj_energy / total_energy).clamp(min=0, max=1)
    return weights, concept_subspace


def compute_domainnet_subspace_weights(
    model,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device,
    n_components: int = 10,
    encoder_type: str = "dino",
):
    """
    Compute DINOv2 PCA subspace weights for DomainNet retain samples.

    forget_loader: DataLoader over the forget domain-class (e.g. sketch-tiger)
    retain_loader: DataLoader over all retain samples
    Returns: weights (N_retain,), subspace (k, D), global_mean (D,)
    """
    model.eval()
    tag = encoder_type.upper()

    print(f"  {tag}: extracting forget features ({len(forget_loader.dataset)} samples)...")
    forget_feat = _extract_features_from_loader(model, forget_loader, device, encoder_type)

    print(f"  {tag}: extracting retain features ({len(retain_loader.dataset)} samples)...")
    retain_feat = _extract_features_from_loader(model, retain_loader, device, encoder_type)

    weights, subspace, global_mean = subspace_weights_from_features(
        forget_feat, retain_feat, n_components
    )

    print(f"  weights — mean={weights.mean():.4f}  max={weights.max():.4f}"
          f"  %>0.05: {(weights > 0.05).float().mean():.1%}")

    return weights.cpu(), subspace.cpu(), global_mean.cpu()


def compute_clip_subspace_weights(
    model,
    preprocess,
    raw_data,
    forget_indices,
    retain_indices,
    device,
    n_components: int = 10,
    encoder_type: str = "clip",
    **kwargs,
):
    """PCA subspace generalization weights using any frozen encoder."""
    model.eval()
    tag = encoder_type.upper()

    print(f"  {tag}: extracting {len(forget_indices)} forget features...")
    forget_feat = _extract_features(model, preprocess, raw_data, forget_indices, device, encoder_type)

    print(f"  {tag}: extracting {len(retain_indices)} retain features...")
    retain_feat = _extract_features(model, preprocess, raw_data, retain_indices, device, encoder_type)

    global_mean = torch.cat([forget_feat, retain_feat], dim=0).mean(dim=0)

    forget_centered = forget_feat - global_mean.unsqueeze(0)
    _, _, Vt = torch.linalg.svd(forget_centered, full_matrices=False)
    subspace = Vt[:n_components]  # (k, D)

    retain_centered = retain_feat - global_mean.unsqueeze(0)
    proj_energy  = (retain_centered @ subspace.T).pow(2).sum(dim=1)
    total_energy = retain_centered.pow(2).sum(dim=1).clamp(min=1e-8)
    weights      = (proj_energy / total_energy).clamp(min=0, max=1)

    print(f"  weights — mean={weights.mean():.4f}  max={weights.max():.4f}"
          f"  %>0.05: {(weights > 0.05).float().mean():.1%}")

    return weights.cpu(), subspace.cpu(), global_mean.cpu()


def compute_clip_raw_similarity_weights(
    model,
    preprocess,
    raw_data,
    forget_indices,
    retain_indices,
    device,
    encoder_type: str = "clip",
    **kwargs,
):
    """Centered cosine similarity weighting (no PCA)."""
    model.eval()

    forget_feat = _extract_features(model, preprocess, raw_data, forget_indices, device, encoder_type)
    retain_feat = _extract_features(model, preprocess, raw_data, retain_indices, device, encoder_type)

    global_mean = torch.cat([forget_feat, retain_feat], dim=0).mean(dim=0)
    proto = F.normalize(forget_feat.mean(dim=0) - global_mean, dim=0)
    retain_centered = F.normalize(retain_feat - global_mean.unsqueeze(0), dim=-1)
    weights = (retain_centered @ proto).clamp(min=0)

    print(f"  weights — mean={weights.mean():.4f}  max={weights.max():.4f}"
          f"  %>0.05: {(weights > 0.05).float().mean():.1%}")

    return weights.cpu(), None, global_mean.cpu()


def compute_random_subspace_weights(
    model,
    preprocess,
    raw_data,
    forget_indices,
    retain_indices,
    device,
    n_components: int = 10,
    encoder_type: str = "clip",
    seed: int = 0,
    **kwargs,
):
    """Random orthogonal subspace — negative control."""
    model.eval()

    forget_feat = _extract_features(model, preprocess, raw_data, forget_indices, device, encoder_type)
    retain_feat = _extract_features(model, preprocess, raw_data, retain_indices, device, encoder_type)

    global_mean = torch.cat([forget_feat, retain_feat], dim=0).mean(dim=0)

    D = forget_feat.shape[1]
    torch.manual_seed(seed)
    rand_mat  = torch.randn(D, n_components, device=device)
    subspace, _ = torch.linalg.qr(rand_mat)
    subspace  = subspace.T  # (k, D)

    retain_centered = retain_feat - global_mean.unsqueeze(0)
    proj_energy  = (retain_centered @ subspace.T).pow(2).sum(dim=1)
    total_energy = retain_centered.pow(2).sum(dim=1).clamp(min=1e-8)
    weights      = (proj_energy / total_energy).clamp(min=0, max=1)

    print(f"  weights — mean={weights.mean():.4f}  max={weights.max():.4f}"
          f"  %>0.05: {(weights > 0.05).float().mean():.1%}")

    return weights.cpu(), subspace.cpu(), global_mean.cpu()


class WeightedSubset(Dataset):
    """Wraps an IndexedSubset to also return a pre-computed per-sample scalar weight."""

    def __init__(self, subset: Dataset, weights: torch.Tensor):
        self.subset  = subset
        self.weights = weights  # (N,)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        return x, y, self.weights[idx].item()
