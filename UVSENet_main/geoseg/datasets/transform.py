import torch
import random
import numpy as np

# ---------------------- Compose ----------------------
class Compose(object):
    """Compose multiple augmentations for img, embed, mask."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, embed, mask):
        """
        img: (C,H,W) 原始影像或输入特征
        embed: (C,H,W) embedding 特征
        mask: (H,W)
        """
        assert img.shape[1:] == mask.shape, f"Image and mask shapes mismatch: {img.shape}, {mask.shape}"
        if embed is not None:
            assert embed.shape[1:] == mask.shape, f"Embed and mask shapes mismatch: {embed.shape}, {mask.shape}"

        for t in self.transforms:
            img, embed, mask = t(img, embed, mask)
        return img, embed, mask

# ---------------------- Basic transforms ----------------------
class RandomCrop(object):
    def __init__(self, size=(256,256)):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, embed, mask):
        C, H, W = img.shape
        th, tw = self.size
        if H <= th and W <= tw:
            return img, embed, mask
        x1 = random.randint(0, max(0, W - tw))
        y1 = random.randint(0, max(0, H - th))
        img = img[:, y1:y1+th, x1:x1+tw]
        if embed is not None:
            embed = embed[:, y1:y1+th, x1:x1+tw]
        mask = mask[y1:y1+th, x1:x1+tw]
        return img, embed, mask

class PadImage(object):
    def __init__(self, size=(256,256), pad_value=0, ignore_index=0):
        self.size = size
        self.pad_value = pad_value
        self.ignore_index = ignore_index

    def __call__(self, img, embed, mask):
        C, H, W = img.shape
        th, tw = self.size
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=self.pad_value)
            if embed is not None:
                embed = torch.nn.functional.pad(embed, (0, pad_w, 0, pad_h), value=self.pad_value)
            mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), value=self.ignore_index)
        return img, embed, mask

class HorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, embed, mask):
        if random.random() < self.prob:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[1])
            if embed is not None:
                embed = torch.flip(embed, dims=[2])
        return img, embed, mask

class VerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, embed, mask):
        if random.random() < self.prob:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[0])
            if embed is not None:
                embed = torch.flip(embed, dims=[1])
        return img, embed, mask

class Resize(object):
    """Resize img and mask to given size (H, W)"""
    def __init__(self, size=(256,256)):
        self.size = size

    def __call__(self, img, embed, mask):
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        if embed is not None:
            embed = torch.nn.functional.interpolate(embed.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.size, mode='nearest').squeeze(0).squeeze(0).long()
        return img, embed, mask

# ---------------------- Embedding-specific augmentation ----------------------
class AddGaussianNoise(object):
    """Add Gaussian noise to embedding features."""
    def __init__(self, mean=0.0, std=0.01, prob=0.5):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, img, embed, mask):
        if embed is not None and random.random() < self.prob:
            noise = torch.randn_like(embed) * self.std + self.mean
            embed = embed + noise
        return img, embed, mask

# ---------------------- SmartCrop ----------------------
class SmartCrop(object):
    """Crop img and mask to ensure no single class dominates too much."""
    def __init__(self, crop_size=(256,256), ignore_index=0, max_ratio=0.75, num_attempts=10):
        self.crop_size = crop_size
        self.ignore_index = ignore_index
        self.max_ratio = max_ratio
        self.num_attempts = num_attempts
        self.random_crop = RandomCrop(crop_size)

    def __call__(self, img, embed, mask):
        for _ in range(self.num_attempts):
            img_crop, embed_crop, mask_crop = self.random_crop(img, embed, mask)
            labels, counts = torch.unique(mask_crop[mask_crop != self.ignore_index], return_counts=True)
            if len(counts) == 0:
                continue
            if counts.max() / counts.sum() < self.max_ratio:
                return img_crop, embed_crop, mask_crop
        return img_crop, embed_crop, mask_crop  # fallback

# ---------------------- Example usage ----------------------
if __name__ == "__main__":
    img = torch.randn(3, 512, 512)       # RGB
    embed = torch.randn(64, 512, 512)    # embedding
    mask = torch.randint(0, 8, (512,512))

    transform = Compose([
        SmartCrop(crop_size=(256,256), ignore_index=12),
        HorizontalFlip(prob=0.5),
        VerticalFlip(prob=0.5),
        AddGaussianNoise(std=0.02),
        PadImage(size=(256,256), pad_value=0, ignore_index=12),
    ])

    img_aug, embed_aug, mask_aug = transform(img, embed, mask)
    print("Shapes after augmentation:", img_aug.shape, embed_aug.shape, mask_aug.shape)
