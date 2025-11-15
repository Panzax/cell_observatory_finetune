# Config Flow Explanation: Swin-UNETR

## Overview
Your config files work by **composing** (merging) many small config files together. This document traces exactly what happens.

---

## File Locations

```
configs/
├── experiments/abc/3D/swin-unetr/
│   └── swin_unetr_experiment.yaml          ← YOUR EXPERIMENT FILE
├── models/swin-unetr/
│   └── swin_unetr_base.yaml                ← YOUR MODEL FILE
├── tasks/
│   └── channel_split.yaml                  ← EXISTING (defines task parameters)
├── clusters/
│   └── local.yaml                          ← EXISTING (GPUs, CPUs, memory)
├── datasets/
│   ├── pretrain_dataset_ray.yaml           ← EXISTING (dataset loader)
│   └── preprocessor/
│       └── channel_split_preprocessor.yaml ← EXISTING (data preprocessing)
├── optimizers/
│   └── lamb.yaml                           ← EXISTING (LAMB optimizer config)
├── schedulers/
│   └── test_warmup_cosine_decay.yaml       ← EXISTING (learning rate schedule)
└── ... (many more existing configs)
```

---

## Step-by-Step: What Happens When You Run the Experiment

### 1. Hydra Loads Your Experiment File
```yaml
# swin_unetr_experiment.yaml
defaults:
  - /tasks/channel_split                    # ← Line 1
  - /paths/abc                              # ← Line 2
  - /clusters/local                         # ← Line 3
  ...
  - /models/swin-unetr/swin_unetr_base      # ← Line 9
  ...
  - _self_                                  # ← Line 19 (this file's values)
```

### 2. Hydra Processes Each Default (Top to Bottom)

**Line 1: `/tasks/channel_split`**
- Loads: `configs/tasks/channel_split.yaml`
- Adds to merged config:
```yaml
task: channel_split
input_channels: 1
output_channels: 2
```

**Line 3: `/clusters/local`**
- Loads: `configs/clusters/local.yaml`
- Adds to merged config:
```yaml
clusters:
  batch_size: 128
  worker_nodes: 1
  gpus_per_worker: 1
  cpus_per_gpu: 8
  batch_size_per_gpu: 128  # Computed from batch_size / total_gpus
```

**Line 6: `/datasets/pretrain_dataset_ray`**
- Loads: `configs/datasets/pretrain_dataset_ray.yaml`
- Adds to merged config:
```yaml
datasets:
  batch_size: ${clusters.batch_size_per_gpu}  # ← References clusters config!
  input_shape: [16, 128, 128, 128, 2]
  patch_shape: [4, 16, 16, 16]
  dataset:
    _target_: data.datasets.pretrain_dataset_ray.PretrainDatasourceRay
    ...
```

**Line 9: `/models/swin-unetr/swin_unetr_base` (YOUR MODEL)**
- Loads: `configs/models/swin-unetr/swin_unetr_base.yaml`
- Adds to merged config:
```yaml
model:
  _target_: cell_observatory_finetune.models.meta_arch.swin_unetr.FinetuneSwinUNETR
  input_fmt: ${dataset_layout_order}  # ← Will be resolved later!
  input_shape: ${datasets.input_shape}  # ← References datasets config!
  patch_shape: ${datasets.patch_shape}  # ← References datasets config!
  feature_size: 48
  depths: [2, 2, 2, 2]
  ...
```

**Line 19: `_self_` (Your Experiment File's Values)**
- Applies values from the experiment file itself
- These OVERRIDE any earlier configs:
```yaml
experiment_name: test_swin_unetr_channel_split
dataset_layout_order: TZYXC  # ← This will resolve ${dataset_layout_order}!
quantization: bfloat16
seed: 42
...
```

### 3. Hydra Resolves Variable References

After all configs are merged, Hydra resolves `${...}` references:

```yaml
# Before resolution:
model:
  input_fmt: ${dataset_layout_order}
  input_shape: ${datasets.input_shape}

# After resolution:
model:
  input_fmt: TZYXC                    # ← From experiment file!
  input_shape: [16, 128, 128, 128, 2] # ← From datasets config!
```

### 4. Final Merged Config (Simplified)

```yaml
# Everything merged together:
experiment_name: test_swin_unetr_channel_split
dataset_layout_order: TZYXC
seed: 42

task: channel_split
output_channels: 2

clusters:
  batch_size: 128
  batch_size_per_gpu: 128
  gpus_per_worker: 1

datasets:
  input_shape: [16, 128, 128, 128, 2]
  patch_shape: [4, 16, 16, 16]
  dataset:
    _target_: data.datasets.pretrain_dataset_ray.PretrainDatasourceRay

model:
  _target_: cell_observatory_finetune.models.meta_arch.swin_unetr.FinetuneSwinUNETR
  input_fmt: TZYXC                    # ← Resolved!
  input_shape: [16, 128, 128, 128, 2] # ← Resolved!
  patch_shape: [4, 16, 16, 16]        # ← Resolved!
  feature_size: 48
  depths: [2, 2, 2, 2]

optimizer:
  _target_: torch.optim.LAMB
  lr: 0.001

scheduler:
  _target_: ...warmup_cosine_decay
```

### 5. Training Code Instantiates Everything

```python
# Hydra does this automatically:
model = FinetuneSwinUNETR(
    input_fmt='TZYXC',
    input_shape=[16, 128, 128, 128, 2],
    patch_shape=[4, 16, 16, 16],
    feature_size=48,
    depths=[2, 2, 2, 2],
    ...
)

optimizer = torch.optim.LAMB(model.parameters(), lr=0.001)
dataset = PretrainDatasourceRay(input_shape=[16, 128, 128, 128, 2], ...)
```

---

## Comparison: Your Config vs. Existing MAE Config

### MAE Experiment (Existing)
```yaml
defaults:
  - /models/mae/large              # ← Uses MAE model
  - /models/decoders/linear        # ← MAE needs separate decoder
```

### Your Swin-UNETR Experiment
```yaml
defaults:
  - /models/swin-unetr/swin_unetr_base  # ← Uses Swin-UNETR model
  # No separate decoder needed - built into Swin-UNETR
```

**Key Difference:**
- MAE = Encoder + Separate Decoder
- Swin-UNETR = Encoder + Decoder built together (U-Net architecture)

---

## Where Configurations Come From

| Config Type | Source | Purpose |
|------------|--------|---------|
| `tasks/` | Existing repository | Task definitions (channel_split, upsample_space) |
| `clusters/` | Existing repository | Hardware configs (GPUs, memory, CPUs) |
| `datasets/` | Existing repository | Data loading and preprocessing |
| `optimizers/` | Existing repository | Optimizer configs (LAMB, AdamW) |
| `schedulers/` | Existing repository | Learning rate schedules |
| `models/mae/` | Existing repository | MAE model variants |
| `models/jepa/` | Existing repository | JEPA model variants |
| **`models/swin-unetr/`** | **YOU CREATED** | **Swin-UNETR model variants** |
| **`experiments/.../swin-unetr/`** | **YOU CREATED** | **Your experiment composition** |

---

## Key Insight

**You're NOT copying anything** - you're **referencing and composing** existing configs!

Your files are like a recipe that says:
> "Take the channel_split task, the local cluster setup, the Ray dataset loader, 
> **MY** Swin-UNETR model, the LAMB optimizer, and the cosine decay scheduler, 
> then mix them all together to create my experiment."

The framework handles all the merging and variable resolution automatically!

