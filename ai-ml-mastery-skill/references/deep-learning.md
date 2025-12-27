# Deep Learning Reference

## Table of Contents
1. [PyTorch Essentials](#pytorch-essentials)
2. [TensorFlow/Keras](#tensorflowkeras)
3. [JAX/Flax](#jaxflax)
4. [Training Best Practices](#training-best-practices)
5. [Common Architectures](#common-architectures)
6. [Debugging Deep Learning](#debugging-deep-learning)

---

## PyTorch Essentials

### Modern PyTorch Setup (2024+)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")  # For tensor cores

# Reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Custom Dataset Template

```python
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        transform: Any | None = None,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = self._load_samples(split)
    
    def _load_samples(self, split: str) -> list[dict]:
        # Load file paths, labels, etc.
        ...
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        # Load and process
        if self.transform:
            sample = self.transform(sample)
        return sample
```

### DataLoader Best Practices

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,               # CPU cores for loading
    pin_memory=True,             # Faster GPU transfer
    persistent_workers=True,     # Keep workers alive
    prefetch_factor=2,           # Batches to prefetch per worker
    drop_last=True,              # Consistent batch sizes
)
```

### Modern Model Architecture

```python
class ModernCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Use Sequential for simple blocks
        self.features = nn.Sequential(
            self._conv_block(in_channels, base_dim),
            self._conv_block(base_dim, base_dim * 2),
            self._conv_block(base_dim * 2, base_dim * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_dim * 4, num_classes),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),  # Modern activation
            nn.MaxPool2d(2),
        )
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

### torch.compile (PyTorch 2.0+)

```python
# Compile for speedup (up to 2x faster)
model = torch.compile(model, mode="reduce-overhead")

# Modes:
# - "default": Good balance
# - "reduce-overhead": Best for small models
# - "max-autotune": Slowest compile, fastest runtime
```

### Mixed Precision Training

```python
scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad(set_to_none=True)
    
    with autocast(device_type="cuda", dtype=torch.float16):
        output = model(batch["input"].to(device))
        loss = criterion(output, batch["target"].to(device))
    
    scaler.scale(loss).backward()
    
    # Gradient clipping with scaler
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Checkpointing (Memory Efficient)

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientBlock(nn.Module):
    def __init__(self, ...):
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.use_checkpoint = True
    
    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
```

### Distributed Training (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    model = Model().to(rank)
    model = DDP(model, device_ids=[rank])
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        # Training loop...
    
    cleanup()

# Launch with:
# torchrun --nproc_per_node=4 train.py
```

---

## TensorFlow/Keras

### Modern Keras Setup

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

# GPU memory growth (prevent OOM)
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")
```

### Keras Functional API (Preferred)

```python
def create_model(input_shape: tuple, num_classes: int) -> Model:
    inputs = keras.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv2D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    
    return Model(inputs, outputs, name="custom_cnn")
```

### Custom Training Loop (TensorFlow)

```python
@tf.function
def train_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
```

### tf.data Pipeline

```python
def create_dataset(
    file_paths: list[str],
    batch_size: int = 32,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
```

---

## JAX/Flax

### JAX Fundamentals

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from flax import linen as nn
from flax.training import train_state
import optax

# JIT compilation
@jit
def forward(params, x):
    return model.apply(params, x)

# Vectorization
batched_forward = vmap(forward, in_axes=(None, 0))

# Automatic differentiation
loss_grad_fn = jax.value_and_grad(loss_fn)
```

### Flax Model

```python
class FlaxCNN(nn.Module):
    num_classes: int
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x
```

---

## Training Best Practices

### Learning Rate Scheduling

```python
# OneCycleLR (PyTorch) - Best general-purpose scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,           # Warmup percentage
    anneal_strategy="cos",
)

# Cosine with warmup (manual)
def cosine_warmup_scheduler(step, total_steps, warmup_steps, max_lr, min_lr=0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### Weight Decay and Regularization

```python
# Separate weight decay from bias/norm layers
param_groups = [
    {"params": [p for n, p in model.named_parameters() 
                if "bias" not in n and "norm" not in n],
     "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() 
                if "bias" in n or "norm" in n],
     "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(param_groups, lr=1e-4)
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop
```

### Model Checkpointing

```python
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer = None):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_metric"]
```

---

## Common Architectures

### Residual Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
```

---

## Debugging Deep Learning

### Shape Debugging

```python
# Hook to print shapes during forward pass
def shape_hook(name):
    def hook(module, input, output):
        print(f"{name}: {[tuple(i.shape) for i in input]} -> {tuple(output.shape)}")
    return hook

# Register hooks
for name, module in model.named_modules():
    module.register_forward_hook(shape_hook(name))
```

### Gradient Debugging

```python
# Check for vanishing/exploding gradients
def check_gradients(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-7:
                print(f"⚠️ Vanishing gradient: {name} (norm={grad_norm:.2e})")
            elif grad_norm > 1e3:
                print(f"⚠️ Exploding gradient: {name} (norm={grad_norm:.2e})")
```

### Memory Profiling

```python
# PyTorch memory summary
print(torch.cuda.memory_summary())

# Track memory allocations
torch.cuda.reset_peak_memory_stats()
# ... run training step ...
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### NaN Detection

```python
torch.autograd.set_detect_anomaly(True)  # Enable during debugging only

# Manual NaN check
def check_nan(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")
```
