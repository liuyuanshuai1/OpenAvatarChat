"""Local streaming CV3/vLLM service using the already benchmarked environment."""
import os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')

import base64
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get('AVATAR_TTS_MODEL', '/accel-models/Fun-CosyVoice3-0.5B-2512')
PROMPT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
REFERENCE = '/opt/CosyVoice/asset/zero_shot_prompt.wav'
VOICE = 'avatar_default'
model = None
initial_hop = None
inference_lock = threading.Lock()
requests_lock = threading.Lock()
requests = {}
app = FastAPI(docs_url=None, redoc_url=None)

class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex, max_length=128)

@app.get('/health')
def health():
    return {'ready': model is not None, 'model': 'CosyVoice3', 'backend': 'vllm',
            'sample_rate': model.sample_rate if model else None,
            'active_requests': len(requests), 'cached_voice': VOICE}

@app.post('/cancel/{request_id}')
def cancel(request_id: str):
    with requests_lock:
        event = requests.get(request_id)
        if event:
            event.set()
    return {'cancelled': event is not None}

def packet(data):
    return (json.dumps(data, ensure_ascii=False) + '\n').encode('utf-8')

@app.post('/tts')
def tts(request: SpeechRequest):
    if model is None:
        raise HTTPException(503, 'TTS is not ready')
    if not request.text.strip():
        raise HTTPException(400, 'Empty text')
    event = threading.Event()
    with requests_lock:
        if request.request_id in requests:
            raise HTTPException(409, 'Duplicate request_id')
        if len(requests) >= 4:
            raise HTTPException(429, 'TTS queue is full')
        requests[request.request_id] = event

    def generate():
        from cosyvoice.utils.common import set_all_random_seed
        generator = None
        start = time.perf_counter()
        duration = 0.0
        first = None
        index = 0
        try:
            with inference_lock:
                if event.is_set():
                    return
                model.model.token_hop_len = initial_hop
                set_all_random_seed(42)
                try:
                    generator = model.inference_zero_shot(
                        request.text, PROMPT, REFERENCE,
                        zero_shot_spk_id=VOICE, stream=True)
                    for item in generator:
                        if event.is_set():
                            # Upstream cleans caches only when exhausted. Discard the
                            # current text piece while queued pieces are cancelled.
                            continue
                        audio = item['tts_speech'].detach().float().cpu().numpy().reshape(-1)
                        if not np.isfinite(audio).all():
                            raise RuntimeError('Non-finite TTS audio')
                        if first is None:
                            first = time.perf_counter() - start
                            print(json.dumps({'event': 'first_pcm', 'request_id': request.request_id,
                                              'seconds': first, 'text_chars': len(request.text)}), flush=True)
                        duration += len(audio) / model.sample_rate
                        yield packet({'request_id': request.request_id, 'sample_rate': model.sample_rate,
                                      'channels': 1, 'dtype': 'float32le', 'chunk_index': index,
                                      'audio': base64.b64encode(audio.astype('<f4').tobytes()).decode('ascii')})
                        index += 1
                    yield packet({'request_id': request.request_id, 'done': True,
                                  'cancelled': event.is_set(), 'audio_seconds': duration})
                finally:
                    # A premature close of upstream skips its cache/thread cleanup.
                    # Finish the current piece on disconnect before another request.
                    if generator:
                        try:
                            for _ in generator:
                                pass
                        finally:
                            generator.close()
                            generator = None
        except GeneratorExit:
            event.set()
            raise
        except Exception as exc:
            print(json.dumps({'event': 'error', 'request_id': request.request_id,
                              'error': repr(exc)}), flush=True)
            yield packet({'request_id': request.request_id, 'error': str(exc), 'done': True})
        finally:
            with requests_lock:
                requests.pop(request.request_id, None)
            elapsed = time.perf_counter() - start
            print(json.dumps({'event': 'request_done', 'request_id': request.request_id,
                              'elapsed_s': elapsed, 'audio_s': duration,
                              'rtf': elapsed / duration if duration else None,
                              'cancelled': event.is_set()}), flush=True)

    return StreamingResponse(generate(), media_type='application/x-ndjson',
                             headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

def main():
    global model, initial_hop
    import wetext
    import vllm
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    from cosyvoice.cli.cosyvoice import AutoModel
    resource_path = Path('/opt/accel-build/wetext')
    def local_snapshot(repo_id, **kwargs):
        if repo_id != 'pengzhendong/wetext':
            raise ValueError(repo_id)
        return str(resource_path)
    wetext.Normalizer.__init__.__globals__['snapshot_download'] = local_snapshot
    ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)
    original_args = vllm.EngineArgs
    def engine_args(*args, **kwargs):
        kwargs.update(max_num_seqs=1, max_model_len=2048, max_num_batched_tokens=512)
        return original_args(*args, **kwargs)
    vllm.EngineArgs = engine_args
    print(json.dumps({'event': 'loading', 'model': MODEL_DIR, 'torch': torch.__version__,
                      'vllm': vllm.__version__, 'gpu': torch.cuda.get_device_name(0)}), flush=True)
    model = AutoModel(model_dir=MODEL_DIR, fp16=False, load_vllm=True)
    if not hasattr(model.model.llm, 'vllm'):
        raise RuntimeError('vLLM was not loaded')
    if model.sample_rate != 24000:
        raise RuntimeError(f'Unexpected output sample rate: {model.sample_rate}')
    initial_hop = model.model.token_hop_len
    model.add_zero_shot_spk(PROMPT, REFERENCE, VOICE)
    model.model.token_hop_len = initial_hop
    for _ in model.inference_zero_shot('我们一起测试数字人的语音和画面是否流畅。',
                                     PROMPT, REFERENCE, zero_shot_spk_id=VOICE, stream=True):
        pass
    torch.cuda.synchronize()
    model.model.token_hop_len = initial_hop
    print(json.dumps({'event': 'ready', 'sample_rate': model.sample_rate,
                      'initial_hop': initial_hop, 'physical_gpu': 0}), flush=True)
    uvicorn.run(app, host='127.0.0.1', port=50001, workers=1)

if __name__ == '__main__':
    main()
