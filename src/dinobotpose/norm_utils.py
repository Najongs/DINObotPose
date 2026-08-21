"""Backbone-specific input normalization.

Image normalization is backbone-specific and a train/eval mismatch silently corrupts
results (the SigLIP2 pose bug: SigLIP wants mean=std=0.5 but was fed ImageNet stats).
Instead of hardcoding per-family `if "siglip" in name`, read the backbone's OWN
AutoImageProcessor.image_mean/std. This is correct for any current/future backbone:
  DINOv3   -> ImageNet [0.485,0.456,0.406]/[0.229,0.224,0.225]
  SigLIP2  -> [0.5,0.5,0.5]
  google/vit (supervised) -> [0.5,0.5,0.5]
  MAE      -> ImageNet
"""

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
HALF = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def get_norm_stats(model_name, random_init=False, verbose=True):
    """Return (mean, std) lists for `model_name`'s expected input normalization.

    random_init=True: the backbone has random weights (no meaningful pretrained
    processor), so use the family's canonical scale (google/vit family = 0.5).
    Otherwise read AutoImageProcessor; fall back to a name heuristic on failure.
    """
    if random_init:
        if verbose:
            print(f"==> random-init ({model_name}): using mean=std=0.5 normalization")
        return HALF
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(model_name)
        mean = list(getattr(proc, "image_mean", None) or IMAGENET[0])
        std = list(getattr(proc, "image_std", None) or IMAGENET[1])
        if verbose:
            print(f"==> {model_name}: normalization from AutoImageProcessor mean={mean} std={std}")
        return mean, std
    except Exception as e:
        mean, std = (HALF if ("siglip" in model_name or "/vit-" in model_name) else IMAGENET)
        if verbose:
            print(f"==> {model_name}: AutoImageProcessor unavailable ({e}); fallback mean={mean} std={std}")
        return mean, std
