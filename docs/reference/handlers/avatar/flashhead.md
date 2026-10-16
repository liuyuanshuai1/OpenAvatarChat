# FlashHead 数字人 Handler

集成 [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) 项目的 Lite 模式，基于扩散模型的实时流式说话头生成。支持单张图像驱动，在单 RTX 4090 上可达 96FPS 或 3 路并发 25+ FPS。

## 依赖模型

```bash
uv run scripts/download_models.py --handler flashhead
```

<details>
<summary>手动下载</summary>

* **Soul-AILab/SoulX-FlashHead-1_3B** [HuggingFace](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
```

* **facebook/wav2vec2-base-960h**（与 LAM 共享）
```bash
git clone --depth 1 https://huggingface.co/facebook/wav2vec2-base-960h ./models/wav2vec2-base-960h
```

</details>

## 配置参数

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
  debug: false
```

* **条件图像**：通过 `cond_image_path` 指定人脸图像
* **模型类型**：`lite`（推荐）或 `pro`
* **人脸裁剪**：设置 `use_face_crop: true` 可自动检测并裁剪

> [!Note]
> FlashHead 依赖 `flash-attn`，`install.py` 会自动编译安装。


## 可选独立推理进程（候选，待GPU实测）

```yaml
FlashHead:
  # 保留原有模型、图像与路径配置
  execution_mode: process     # 默认in_process，可用于性能对照
  concurrent_limit: 1
  fps: 25
  algo_audio_sample_rate: 16000
  output_audio_sample_rate: 24000
  max_buffer_seconds: 2.0
  worker_threads: 4
  worker_warmup_batches: 3
  worker_start_timeout: 600
  worker_request_timeout: 60
```

process模式使用spawn子进程独占模型权重，加载并完成真实推理预热后才返回就绪。帧批次通过单槽共享内存传回，父进程保留音视频配对、播放节拍与取消控制。当前仅支持一个活动会话；容器需允许子进程和共享内存，空间至少容纳一批图像（Lite 24×512×512×3约18MiB），并留运行时余量。

输出队列按配对帧背压，容量至少一批。取消代号防止旧任务迟到结果重新入队；RTC清理尚未发送的数据，但无法撤回浏览器已有解码缓冲。worker失败或超时后关闭，不自动重启、不静默回退；恢复需重启服务。RTC慢消费者队列仍需后续媒体时长水位验收，不能据此宣称全链路缓存有界。

配置暂只允许25FPS、16kHz模型输入和24kHz输出；其他输入采样率由handler重采样。切换回in_process仅回退推理位置，其余队列/转换改进仍生效；完整回退应恢复原镜像或源码。上线前应验收固定音频、长播放、频繁打断、断连与worker异常。独立原型的性能数字不等同于此候选已完成GPU验收。

RTX 4090对照时，两组均显式设置 `enable_compile: true`，与已验证隔离原型保持条件一致。兼容性默认值为false；其他GPU先验证编译兼容性。帧在GPU上转为uint8后再回传，真实GPU验收仍待完成。
