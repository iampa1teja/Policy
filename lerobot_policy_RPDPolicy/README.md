# RPDPolicy

A third-party [LeRobot](https://github.com/huggingface/lerobot) robot policy that conditions a
flow-matching action expert on **object-track tokens** — geometry-only bounding-box histories from a
bespoke detector/tracker — in addition to the usual vision tokens and proprioceptive state.

The track tokens carry only geometry (box coordinates over time) and an externally assigned role, never
appearance. The intent is attribute-invariant generalization: a policy trained to manipulate one object
should transfer to a differently-colored or differently-sized object of the same role, because the
conditioning it sees is blind to color and texture.

## Two stacks

The repository holds two packages side by side under `src/`:

| Stack | Path | Role |
|---|---|---|
| `CNN` | `src/CNN/` | Standalone object detection + tracking. Trained **separately, ahead of the policy**. Its checkpoint is a hard prerequisite. |
| `lerobot_policy_RPDPolicy` | `src/lerobot_policy_RPDPolicy/` | The LeRobot policy that consumes the **frozen** `CNN` stack. |

The seam between them is the checkpoint contract (see below). The policy never imports live `CNN` modules
for training weights; it rebuilds the perception halves from `model_config` and loads their state dicts.

## Environment

Use the `lerobot` conda env (`lerobot` 0.6.1, `torch` 2.11.0+cu130, `transformers` 5.15.0,
`ultralytics` 8.4.116, `torchvision` 0.26.0):

```bash
conda activate lerobot
```

`timm` is imported by the perception backbone but not declared anywhere, so a fresh env needs:

```bash
pip install timm
```

The package is not pip-installed and has no build config. Both packages resolve as an implicit namespace
package rooted at `lerobot_policy_RPDPolicy/`, so `src` is part of the import path:

```bash
cd lerobot_policy_RPDPolicy && python -c "import src.lerobot_policy_RPDPolicy as p; print(p.__all__)"
```

## Step 1 — train the perception stack

Detection + tracking is trained in Python, not via a CLI:

```python
from src.CNN import CNN
from src.CNN.core.dataset import Dataset

model = CNN(num_classes=N, backbone_name="resnet50", neck="bifpn", use_cbam=True)
model.fit(Dataset("/path/to/data"), epochs=100, output_dir="./runs")
```

`Dataset` expects a YOLO layout (`root/{train,test}/{images,labels}`, one `.txt` per image with
`class x y w h` normalized). Labels are assumed **1-indexed** on disk by default; pass
`one_indexed_labels=False` to `Dataset` if yours are 0-indexed.

Training writes a checkpoint (`epoch_{n}.pt`) whose contract the policy depends on. `save_checkpoint`
deliberately stores the two perception halves separately so the policy can rebuild each independently:

```
{
  "feature_extractor_state_dict": ...,   # FeatureExtraction weights
  "detector_state_dict": ...,            # Detect head weights
  "model_config": {                      # how to reconstruct the modules
     "backbone_name", "neck", "feature_channels", "bifpn_layers", "use_cbam"
  }
}
```

`RPDConfig.validate_features` enforces exactly these three keys.

## Step 2 — run the policy

```python
from src.lerobot_policy_RPDPolicy import RPDConfig, RPDPolicy, make_RPDPolicy_pre_post_processors

config = RPDConfig(model_checkpoint="runs/epoch_100.pt")
# config.input_features / output_features are supplied by the LeRobot dataset/env wiring.

policy = RPDPolicy(config)
preprocessor, postprocessor = make_RPDPolicy_pre_post_processors(config, dataset_stats)
```

Because registration follows LeRobot's naming convention
(`@PreTrainedConfig.register_subclass("RPDPolicy")`), the standard factory also resolves it:

```python
from lerobot.policies.factory import get_policy_class
get_policy_class("RPDPolicy")   # -> RPDPolicy
```

## Architecture

```
images ─FeatureExtraction─> multi-level features ─VisionTokenizer─> TokenProjector ─┐
                                                                                    ├─> conditions
tracks ─TrackTokenizer──> TokenProjector ───────────────────────────────────────────┘   {name: (weight, embeds)}
                                                                                              │
state ───────────────────────────────────────────────────────────────────> ConditionalFlowMatching
                                                                                              │
                                                              ConditionGate: concat(w_i * cond_i) + mask
                                                                                              │
                                                                       ActionExpert (PiGemmaModel)
                                                                                              │
                                              training: MSE(v_pred, a1 - a0)  │  inference: Euler ODE
```

Key design points:

- **Perception is frozen.** With `freeze_perception=True` (default) the feature extractor runs under
  `no_grad` and its outputs are detached; only the tokenizers, projectors, and action expert train.
- **`ConditionGate` concatenates, it does not sum.** Conditions differ in token count (vision vs. track)
  and share only the hidden dim, so they are concatenated along the sequence axis. Per-condition validity
  masks are passed explicitly, never inferred, so padded track slots stay correct.
- **Ablate by masking whole conditions, not by `condition_weights`.** The scalar weights are applied
  before the action expert's first RMSNorm, which is scale-invariant, so they divide back out and are
  mathematically inert. Use `use_vision_tokens` / `use_track_tokens` to toggle a modality in or out for
  ablations (vision-only, track-only, both).
- **`ActionExpert` layout** is `[gated prefix | state | noisy actions]`; velocity is read off the last
  `horizon` positions. Timestep `t` enters through `SinusoidalTimeEmbedding` → adaptive RMSNorm, not as a
  token. Attention is **blocked, not causal**: prefix+state cannot see the actions, everything else is
  bidirectional. The 4D bool mask is only valid for `sdpa`, which `make_action_expert_config()` pins.
- **`TrackTokenizer` has two paths.** Realtime `forward(tracks)` is stateful and returns
  `dict[track_id, Tensor]`; training `forward_history_batch(boxes, frame_ids, history_mask)` is stateless,
  takes `[B, N, T, 4]`, and runs one packed LSTM over all valid histories. Absolute frame ids are
  converted to age-from-newest before the temporal embedding lookup.
- **`TrackTokenizerProcessorStep` is pure data shaping — it does not tokenize.** Tokenization lives in the
  policy so its parameters receive gradients; a tokenizer inside a processor step never would. Offline the
  step emits `track_boxes` / `track_frame_ids` / `track_history_mask`; realtime it passes the raw frame
  through for the model's stateful tokenizer.

## Configuration

`RPDConfig` (`configuration_RPDPolicy.py`) — notable fields:

| Field | Default | Meaning |
|---|---|---|
| `model_checkpoint` | `None` (required) | Path to the trained CNN checkpoint. |
| `horizon` / `n_action_steps` | `50` / `50` | Predicted chunk length / steps executed per inference. |
| `hidden_dim` | `256` | Shared condition/action-expert hidden dim. |
| `use_vision_tokens` / `use_track_tokens` | `True` / `True` | Ablation toggles for each modality. |
| `freeze_perception` | `True` | Freeze the CNN feature extractor. |
| `max_tracks` / `max_history_len` | `16` / `30` | Track slots and per-track history window. |
| `num_inference_steps` | `10` | Euler ODE steps at inference. |

## Track data format

- **Offline** rows are 9 columns: `[frame_id, x1, y1, x2, y2, track_id, score, cls, idx]`.
- **Realtime** tracker rows are 8 columns with `track_id` at index 4.

The processor dispatches on type: a `list` of per-episode arrays is treated as an offline batch, anything
else as a single realtime frame.

## Status

The policy builds, runs a training forward + backward, and runs inference (`predict_action_chunk` /
`select_action`) end to end against a trained CNN checkpoint. There is no test suite; `src/.../test.py`
(if present, untracked) is a rejected alternative flow-matching design, **not** a test — do not import it.

An offline pipeline that extracts 9-column track rows from a LeRobot dataset is still required before real
training on recorded episodes.
