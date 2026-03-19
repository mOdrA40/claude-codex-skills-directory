# Computer Vision Reference

## Table of Contents
1. [Image Processing Fundamentals](#image-processing-fundamentals)
2. [CNN Architectures](#cnn-architectures)
3. [Object Detection](#object-detection)
4. [Segmentation](#segmentation)
5. [Vision Transformers](#vision-transformers)
6. [GANs & Diffusion](#gans--diffusion)
7. [Data Augmentation](#data-augmentation)
8. [Pretrained Models (timm)](#pretrained-models-timm)

---

## Image Processing Fundamentals

### OpenCV Essentials

```python
import cv2
import numpy as np
from pathlib import Path

def load_image(path: Path, color: bool = True) -> np.ndarray:
    """Load image with proper color conversion."""
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color else img


def resize_with_aspect_ratio(
    img: np.ndarray,
    target_size: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize maintaining aspect ratio."""
    h, w = img.shape[:2]
    if h > w:
        new_h, new_w = target_size, int(w * target_size / h)
    else:
        new_h, new_w = int(h * target_size / w), target_size
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(img)
```

### PIL/Pillow Operations

```python
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io

def smart_resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize with padding to maintain aspect ratio."""
    return ImageOps.pad(img, size, method=Image.Resampling.LANCZOS, color="black")


def convert_to_tensor_format(img: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized tensor format (C, H, W)."""
    arr = np.array(img).astype(np.float32) / 255.0
    if len(arr.shape) == 2:
        arr = arr[np.newaxis, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    return arr


def create_thumbnail(img: Image.Image, max_size: int = 256) -> Image.Image:
    """Create thumbnail preserving aspect ratio."""
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img
```

---

## CNN Architectures

### Modern ConvNeXt Block

```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block - modernized ConvNet design."""
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim), requires_grad=True
        ) if layer_scale_init > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input + self.drop_path(x)


class DropPath(nn.Module):
    """Stochastic Depth - drop entire residual branch."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor
```

### EfficientNet-style MBConv

```python
class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (EfficientNet)."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        ])
        
        # Squeeze-and-Excitation
        if se_ratio:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(SEModule(hidden_dim, se_channels))
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""
    def __init__(self, channels: int, se_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s
```

---

## Object Detection

### YOLO-style Detection Head

```python
class YOLOHead(nn.Module):
    """YOLO detection head."""
    def __init__(
        self,
        num_classes: int,
        anchors: list[tuple[int, int]],
        in_channels: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.register_buffer("anchors", torch.tensor(anchors).float())
        
        # Output: (x, y, w, h, objectness, class_probs)
        out_channels = self.num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels * 2, out_channels, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        pred = self.conv(x)
        pred = pred.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        pred = pred.permute(0, 1, 3, 4, 2)  # (B, A, H, W, 5+C)
        return pred


def decode_yolo_output(
    pred: torch.Tensor,
    anchors: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Decode YOLO predictions to bounding boxes."""
    B, A, H, W, _ = pred.shape
    device = pred.device
    
    # Grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    
    # Decode
    xy = (pred[..., :2].sigmoid() + torch.stack([grid_x, grid_y], dim=-1)) * stride
    wh = pred[..., 2:4].exp() * anchors.view(1, A, 1, 1, 2)
    conf = pred[..., 4:5].sigmoid()
    cls = pred[..., 5:].sigmoid()
    
    return torch.cat([xy, wh, conf, cls], dim=-1)
```

### Non-Maximum Suppression

```python
def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Standard NMS implementation."""
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """NMS per class."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Offset boxes by class to prevent cross-class suppression
    max_coordinate = boxes.max()
    offsets = classes.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    return nms(boxes_for_nms, scores, iou_threshold)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-NMS with Gaussian penalty."""
    keep = []
    indices = torch.argsort(scores, descending=True)
    
    while indices.numel() > 0:
        i = indices[0]
        keep.append(i)
        
        if indices.numel() == 1:
            break
        
        indices = indices[1:]
        ious = box_iou(boxes[i:i+1], boxes[indices])[0]
        
        # Gaussian penalty
        weights = torch.exp(-(ious ** 2) / sigma)
        scores[indices] *= weights
        
        # Remove low-scoring boxes
        mask = scores[indices] > score_threshold
        indices = indices[mask]
    
    return torch.stack(keep), scores[keep]
```

### Using Ultralytics YOLOv8

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")  # nano, small, medium, large, xlarge

# Train
results = model.train(
    data="coco128.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
)

# Inference
results = model("image.jpg")
boxes = results[0].boxes.xyxy  # (x1, y1, x2, y2)
classes = results[0].boxes.cls
confidences = results[0].boxes.conf

# Export
model.export(format="onnx")
```

---

## Segmentation

### U-Net Architecture

```python
class UNet(nn.Module):
    """U-Net for semantic segmentation."""
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: list[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        
        return self.final_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
```

### Segmentation Losses

```python
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.sigmoid()
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classes."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss."""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)
```

---

## Vision Transformers

### ViT Patch Embedding

```python
class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        return self.head(x[:, 0])  # CLS token only
```

---

## GANs & Diffusion

### Modern GAN Discriminator

```python
class Discriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, in_channels: int = 3, features: list[int] = [64, 128, 256, 512]):
        super().__init__()
        layers = []
        for i, feature in enumerate(features):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if i == 0 else features[i-1],
                        feature,
                        kernel_size=4,
                        stride=2 if i < len(features) - 1 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(feature) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, padding=1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

### Diffusion U-Net Components

```python
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
```

---

## Data Augmentation

### Albumentations (Recommended)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50)),
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=7),
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.CLAHE(clip_limit=4.0),
    ], p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

### MixUp and CutMix

```python
def mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """MixUp augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = images.shape
    index = torch.randperm(batch_size, device=images.device)
    
    # Random box
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    
    return images, labels, labels[index], lam
```

---

## Pretrained Models (timm)

```python
import timm

# List available models
timm.list_models("*efficientnet*")

# Load pretrained
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=10,
)

# Feature extraction
model = timm.create_model(
    "resnet50",
    pretrained=True,
    num_classes=0,  # Remove classifier
    global_pool="",  # Remove global pooling
)

# Get model config
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
```
