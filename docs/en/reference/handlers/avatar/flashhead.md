# FlashHead Avatar Handler

Integrates [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) Lite mode. Achieves 96 FPS on RTX 4090 or 3 concurrent sessions at 25+ FPS.

## Model Dependencies

```bash
uv run scripts/download_models.py --handler flashhead
```

## Configuration

```yaml
FlashHead:
  module: avatar/flashhead/avatar_handler_flashhead
  ckpt_dir: "models/SoulX-FlashHead-1_3B"
  wav2vec_dir: "models/wav2vec2-base-960h"
  model_type: "lite"
  cond_image_path: "resource/avatar/flashhead/default.png"
  fps: 25
  base_seed: 42
  use_face_crop: false
```

> [!Note]
> FlashHead depends on `flash-attn`. `install.py` compiles it automatically.


## Optional inference process (candidate; GPU validation pending)

Set `execution_mode: process` on FlashHead; the default remains `in_process`.
Process mode requires `concurrent_limit: 1`, 25 FPS, 16000 Hz algorithm input and
24000 Hz playback output. Incoming TTS audio is resampled as needed.

Options: `max_buffer_seconds: 2.0`, `worker_threads: 4`,
`worker_warmup_batches: 3`, `worker_start_timeout: 600`,
`worker_request_timeout: 60` (timeouts in seconds).

A spawned worker owns the CUDA weights and performs real inference warmup before
load completes. One shared-memory batch slot transfers BGR frames; the parent
copies each batch before reuse and maintains paired audio/video backpressure.
Containers must permit child processes and have shared-memory capacity for at
least one batch (about 18 MiB for 24 Lite frames at 512x512), plus runtime overhead.

Cancellation generations discard late results. RTC cancellation clears queued
media, but cannot retract the browser decoder buffer. Worker errors/timeouts fail
closed; there is no automatic restart or fallback. Restart the service to recover.
The RTC slow-consumer queue still needs media-duration limits; this change does
not claim end-to-end bounded buffering. Switching to in_process changes only the
inference location; restore the previous source/image for a complete rollback.
Validate fixed audio, long playback, interruptions, reconnects and worker failures
on the target GPU before deployment. Earlier prototype benchmarks are not a GPU
acceptance result for this candidate.

For the RTX 4090 comparison, explicitly set `enable_compile: true` in both modes to match the validated prototype. The compatibility default is false; real GPU acceptance remains pending. Frames are quantized to uint8 before host transfer.
