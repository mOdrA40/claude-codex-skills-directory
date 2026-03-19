# Transformers & LLM Reference

## Table of Contents
1. [Transformer Architecture](#transformer-architecture)
2. [HuggingFace Transformers](#huggingface-transformers)
3. [Fine-tuning Strategies](#fine-tuning-strategies)
4. [PEFT Methods (LoRA, QLoRA)](#peft-methods)
5. [LLM Inference Optimization](#llm-inference-optimization)
6. [Tokenization](#tokenization)
7. [Evaluation & Benchmarking](#evaluation--benchmarking)

---

## Transformer Architecture

### Complete Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm (modern) vs Post-norm (original)
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.W_qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # Use Flash Attention if available
        if hasattr(F, "scaled_dot_product_attention"):
            attn_mask = None if mask is None else mask.bool()
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = self.dropout(attn.softmax(dim=-1))
            out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.W_o(out)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward (modern LLMs like Llama)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU has 3 projections
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE - Used in Llama, Mistral, etc."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        
        return x * cos + rotated * sin
```

### Causal Language Model

```python
class CausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(d_model // num_heads, max_seq_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).bool()
        
        x = self.token_emb(input_ids)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(input_ids)["logits"][:, -1, :]
            
            # Temperature scaling
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
```

---

## HuggingFace Transformers

### Loading Models

```python
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch

# Standard loading
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # Requires flash-attn
)

# 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token  # Important for batch training
```

### Fine-tuning with Trainer

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 4 * 8 = 32
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
```

---

## PEFT Methods

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling factor
    target_modules=[               # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

### QLoRA (Quantized LoRA)

```python
from peft import prepare_model_for_kbit_training

# After loading with BitsAndBytesConfig
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Training with QLoRA typically uses:
# - Lower learning rate (1e-4 to 2e-4)
# - paged_adamw_8bit optimizer
# - gradient checkpointing
```

### Merging LoRA Weights

```python
# After training, merge LoRA weights back into base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")

# Or save only LoRA weights (smaller)
model.save_pretrained("./lora_weights")

# Load LoRA weights later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
model = PeftModel.from_pretrained(base_model, "./lora_weights")
```

---

## LLM Inference Optimization

### vLLM (Production Serving)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,  # Multi-GPU
    dtype="bfloat16",
    max_model_len=4096,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

outputs = llm.generate(prompts, sampling_params)
```

### KV Cache Optimization

```python
class KVCache:
    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int):
        self.cache_k = torch.zeros(max_batch_size, max_seq_len, n_heads, head_dim)
        self.cache_v = torch.zeros(max_batch_size, max_seq_len, n_heads, head_dim)
        self.seq_len = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _, _ = k.shape
        self.cache_k[:batch_size, self.seq_len:self.seq_len + seq_len] = k
        self.cache_v[:batch_size, self.seq_len:self.seq_len + seq_len] = v
        self.seq_len += seq_len
        return self.cache_k[:batch_size, :self.seq_len], self.cache_v[:batch_size, :self.seq_len]
```

### Speculative Decoding

```python
def speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,  # Draft tokens per step
):
    """Generate tokens using speculative decoding for speedup."""
    for _ in range(max_new_tokens // gamma):
        # Generate gamma draft tokens
        draft_tokens = []
        draft_input = input_ids
        for _ in range(gamma):
            draft_logits = draft_model(draft_input)["logits"][:, -1]
            token = draft_logits.argmax(dim=-1, keepdim=True)
            draft_tokens.append(token)
            draft_input = torch.cat([draft_input, token], dim=1)
        
        # Verify with target model
        target_logits = target_model(draft_input)["logits"]
        
        # Accept/reject draft tokens
        accepted = 0
        for i, draft_token in enumerate(draft_tokens):
            pos = input_ids.shape[1] + i
            if target_logits[:, pos].argmax(dim=-1) == draft_token.squeeze():
                accepted += 1
            else:
                break
        
        # Update input_ids with accepted tokens + 1 from target
        input_ids = torch.cat([
            input_ids,
            torch.stack(draft_tokens[:accepted], dim=1).squeeze(-1),
            target_logits[:, input_ids.shape[1] + accepted].argmax(dim=-1, keepdim=True),
        ], dim=1)
    
    return input_ids
```

---

## Tokenization

### Custom Tokenizer Training

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# BPE Tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
    min_frequency=2,
)

tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

### Handling Long Sequences

```python
def chunk_and_tokenize(
    text: str,
    tokenizer,
    max_length: int = 2048,
    stride: int = 512,
) -> list[dict]:
    """Tokenize long text with sliding window."""
    tokens = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_offsets_mapping=True,
    )
    return tokens
```

---

## Evaluation & Benchmarking

### Perplexity

```python
@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str, stride: int = 512) -> float:
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.max_position_embeddings
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    return torch.exp(torch.stack(nlls).sum() / end_loc).item()
```

### BLEU, ROUGE, BERTScore

```python
from evaluate import load

# BLEU
bleu = load("bleu")
results = bleu.compute(predictions=["hello world"], references=[["hello world"]])

# ROUGE
rouge = load("rouge")
results = rouge.compute(predictions=predictions, references=references)

# BERTScore
bertscore = load("bertscore")
results = bertscore.compute(predictions=predictions, references=references, lang="en")
```

### LM Evaluation Harness

```bash
# Evaluate on common benchmarks
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks hellaswag,mmlu,arc_challenge \
    --batch_size 8 \
    --output_path ./results
```
