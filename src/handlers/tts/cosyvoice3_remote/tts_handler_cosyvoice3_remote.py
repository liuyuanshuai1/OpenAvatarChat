"""Stream PCM from the local GPU0 CosyVoice3/vLLM service."""
import base64
import json
import queue
import re
import threading
import time
import uuid
from typing import Optional

import numpy as np
import requests
from loguru import logger
from pydantic import Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import SignalFilterRule
from chat_engine.data_models.chat_signal_type import ChatSignalType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


class RemoteTTSConfig(HandlerBaseConfigModel):
    api_url: str = 'http://127.0.0.1:50001'
    sample_rate: int = 24000
    min_text_chars: int = Field(default=8, ge=1, le=80)
    max_text_chars: int = Field(default=80, ge=8, le=500)
    read_timeout: float = Field(default=60, ge=5, le=180)
    startup_timeout: float = Field(default=300, ge=5, le=600)


def split_ready_text(text, min_chars=8, max_chars=80, final=False):
    """Keep short greetings with the following sentence; bound long buffers."""
    ready = []
    while text:
        ends = [m.end() for m in re.finditer(r'[。！？!?；;\n]', text)
                if min_chars <= m.end() <= max_chars]
        if ends:
            cut = ends[0]
        elif len(text) >= max_chars:
            cuts = [m.end() for m in re.finditer(r'[,， ]', text[:max_chars])
                    if m.end() >= min_chars]
            cut = cuts[-1] if cuts else max_chars
        elif final:
            cut = len(text)
        else:
            break
        piece, text = text[:cut].strip(), text[cut:]
        if piece:
            ready.append(piece)
    return ready, text


class RemoteTTSContext(HandlerContext):
    def __init__(self, session_id):
        super().__init__(session_id)
        self.pending = ''
        self.input_key = None
        self.blocked_input_key = None
        self.generation = 0
        self.jobs = queue.Queue()
        self.stop = threading.Event()
        self.response = None
        self.request_id = None
        self.worker = None
        self.state_lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.cancel_pending = {}
        self.cancel_worker = None


class HandlerTTSRemote(HandlerBase):
    def __init__(self):
        super().__init__()
        self.config = None
        self.definition = None

    def get_handler_info(self):
        return HandlerBaseInfo(name='CosyVoice3', config_model=RemoteTTSConfig)

    def load(self, engine_config, handler_config: Optional[RemoteTTSConfig] = None):
        self.config = handler_config or RemoteTTSConfig()
        if self.config.min_text_chars > self.config.max_text_chars:
            raise ValueError('min_text_chars exceeds max_text_chars')
        with requests.Session() as client:
            client.trust_env = False
            deadline = time.monotonic() + self.config.startup_timeout
            last_error = None
            while time.monotonic() < deadline:
                try:
                    response = client.get(self.config.api_url.rstrip('/') + '/health', timeout=(3, 5))
                    response.raise_for_status()
                    health = response.json()
                    if health.get('ready'):
                        break
                except requests.RequestException as exc:
                    last_error = type(exc).__name__
                time.sleep(1)
            else:
                raise RuntimeError(f'CosyVoice3 service startup timeout: {last_error}')
        if not health.get('ready') or health.get('backend') != 'vllm':
            raise RuntimeError('CosyVoice3/vLLM service is not ready')
        if health.get('sample_rate') != self.config.sample_rate:
            raise RuntimeError('TTS sample rate does not match output definition')
        self.definition = DataBundleDefinition()
        self.definition.add_entry(DataBundleEntry.create_audio_entry('avatar_audio', 1, self.config.sample_rate))
        self.definition.lockdown()
        logger.info('CosyVoice3 remote ready: url={}, sample_rate={}, cached_voice={}',
                    self.config.api_url, self.config.sample_rate, health.get('cached_voice'))

    def create_context(self, session_context, handler_config=None):
        return RemoteTTSContext(session_context.session_info.session_id)

    def get_handler_detail(self, session_context, context):
        return HandlerDetail(
            inputs={ChatDataType.AVATAR_TEXT: HandlerDataInfo(type=ChatDataType.AVATAR_TEXT)},
            outputs={ChatDataType.AVATAR_AUDIO: HandlerDataInfo(type=ChatDataType.AVATAR_AUDIO,
                                                              definition=self.definition)},
            signal_filters=[SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, ChatDataType.AVATAR_AUDIO),
                            SignalFilterRule(ChatSignalType.INTERRUPT, None, None)])

    def start_context(self, session_context, context):
        context.cancel_worker = threading.Thread(
            target=self._cancel_worker, args=(context,), daemon=True,
            name='CosyVoice3-Control-' + context.session_id)
        context.cancel_worker.start()
        context.worker = threading.Thread(target=self._worker, args=(context,), daemon=True,
                                          name='CosyVoice3-' + context.session_id)
        context.worker.start()

    def _cancel(self, context):
        begin = time.monotonic()
        with context.state_lock:
            context.generation += 1
            context.blocked_input_key = context.input_key
            context.pending = ''
            while True:
                try:
                    context.jobs.get_nowait()
                except queue.Empty:
                    break
            # Capture this request's response. Never close context.response later:
            # by then it could belong to the next generation.
            if context.request_id:
                context.cancel_pending[context.request_id] = context.response
                context.cancel_event.set()
        logger.info('TTS_CANCEL_CALLBACK session={} elapsed_ms={:.3f}',
                    context.session_id, (time.monotonic() - begin) * 1000)

    def _cancel_worker(self, context):
        """Network I/O and requests.close must never block signal distribution."""
        with requests.Session() as client:
            client.trust_env = False
            while True:
                context.cancel_event.wait(.2)
                with context.state_lock:
                    pending = context.cancel_pending
                    context.cancel_pending = {}
                    context.cancel_event.clear()
                    finished = context.stop.is_set() and not pending
                if finished:
                    return
                for request_id, response in pending.items():
                    begin = time.monotonic()
                    try:
                        client.post(self.config.api_url.rstrip('/') + '/cancel/' + request_id,
                                    timeout=(1, 2))
                    except requests.RequestException:
                        logger.debug('TTS cancellation endpoint unavailable for {}', request_id)
                    finally:
                        if response is not None:
                            try:
                                response.close()
                            except Exception:
                                pass
                    logger.info('TTS_CANCEL_CONTROL session={} request={} elapsed_ms={:.3f}',
                                context.session_id, request_id, (time.monotonic() - begin) * 1000)

    def handle(self, context, inputs, output_definitions):
        with context.state_lock:
            self._handle_locked(context, inputs, output_definitions)

    def _handle_locked(self, context, inputs, output_definitions):
        key = inputs.stream_id.stream_key_str if inputs.stream_id else 'default'
        if key != context.input_key:
            self._cancel(context)
            context.input_key = key
            context.blocked_input_key = None
        if key == context.blocked_input_key:
            return
        value = inputs.data.get_main_data()
        if value:
            value = re.sub(r'<\|.*?\|>', '', str(value))
            context.pending += re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff,.~!?，。！？；;：:\n ]', '', value)
        pieces, context.pending = split_ready_text(
            context.pending, self.config.min_text_chars, self.config.max_text_chars,
            final=inputs.is_last_data)
        for text in pieces:
            logger.info('CosyVoice3 text queued: session={}, chars={}, text={}', context.session_id, len(text), text)
            context.jobs.put((context.generation, text, time.monotonic()))
        if inputs.is_last_data:
            context.jobs.put((context.generation, None, time.monotonic()))

    def _emit(self, context, generation, audio, finish=False):
        with context.state_lock:
            if context.stop.is_set() or context.generation != generation:
                return
            bundle = DataBundle(self.definition)
            bundle.set_main_data(np.asarray(audio, dtype=np.float32).reshape(1, -1))
            context.submit_data(bundle, finish_stream=finish)

    def _worker(self, context):
        with requests.Session() as client:
            client.trust_env = False
            while not context.stop.is_set():
                job = context.jobs.get()
                if job is None:
                    break
                generation, text, queued = job
                if generation != context.generation:
                    continue
                if text is None:
                    self._emit(context, generation, np.zeros(240, dtype=np.float32), finish=True)
                    continue
                request_id = uuid.uuid4().hex
                with context.state_lock:
                    if generation != context.generation or context.stop.is_set():
                        continue
                    context.request_id = request_id
                begin = time.monotonic()
                first = True
                done = False
                try:
                    with client.post(self.config.api_url.rstrip('/') + '/tts',
                                     json={'text': text, 'request_id': request_id}, stream=True,
                                     timeout=(3, self.config.read_timeout)) as response:
                        with context.state_lock:
                            context.response = response
                            # Cancel may have arrived before POST registered the
                            # request. Re-notify now that the response exists.
                            if generation != context.generation or context.stop.is_set():
                                context.cancel_pending[request_id] = response
                                context.cancel_event.set()
                        response.raise_for_status()
                        for line in response.iter_lines(chunk_size=16384):
                            if generation != context.generation or context.stop.is_set():
                                break
                            if not line:
                                continue
                            item = json.loads(line)
                            if item.get('request_id') != request_id:
                                raise ValueError('TTS request_id mismatch')
                            if item.get('error'):
                                raise RuntimeError(item['error'])
                            if item.get('done'):
                                done = True
                                break
                            if item.get('sample_rate') != self.config.sample_rate or item.get('dtype') != 'float32le' or item.get('channels') != 1:
                                raise ValueError('Unexpected PCM format')
                            audio = np.frombuffer(base64.b64decode(item['audio'], validate=True), dtype='<f4').copy()
                            if not audio.size or not np.isfinite(audio).all():
                                raise ValueError('Invalid PCM chunk')
                            if first:
                                logger.info('CosyVoice3 first PCM: session={}, request={}, elapsed={:.3f}s, samples={}',
                                            context.session_id, request_id, time.monotonic() - begin, len(audio))
                                first = False
                            self._emit(context, generation, audio)
                    if not done and generation == context.generation and not context.stop.is_set():
                        raise RuntimeError('TTS stream ended before done marker')
                except Exception as exc:
                    if generation == context.generation and not context.stop.is_set():
                        logger.error('CosyVoice3 request failed: session={}, request={}, error={}',
                                     context.session_id, request_id, repr(exc))
                finally:
                    with context.state_lock:
                        if context.request_id == request_id:
                            context.response = None
                            context.request_id = None

    def on_signal(self, context, signal):
        if signal.type == ChatSignalType.INTERRUPT:
            self._cancel(context)
        elif signal.type == ChatSignalType.STREAM_CANCEL and signal.related_stream:
            if signal.related_stream.producer_name == context.owner:
                self._cancel(context)

    def destroy_context(self, context):
        context.stop.set()
        self._cancel(context)
        context.jobs.put(None)
        context.cancel_event.set()
        if context.worker:
            context.worker.join(timeout=3)
        if context.cancel_worker:
            context.cancel_worker.join(timeout=3)


handler_class = HandlerTTSRemote
