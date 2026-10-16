import asyncio
import threading
import time
from types import SimpleNamespace as NS

import numpy as np

from handlers.client.rtc_client.thread_safe_queue import ThreadSafeAsyncQueue
from handlers.tts.cosyvoice3_remote.tts_handler_cosyvoice3_remote import (
    HandlerTTSRemote, RemoteTTSContext, RemoteTTSConfig,
)


def test_cancel_does_not_block_signal_distribution(monkeypatch):
    from chat_engine.core.signal_manager import SignalManager
    from chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
    entered, release, next_listener = threading.Event(), threading.Event(), threading.Event()
    class Client:
        trust_env = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, *args, **kwargs):
            entered.set()
            release.wait(2)
    monkeypatch.setattr('handlers.tts.cosyvoice3_remote.tts_handler_cosyvoice3_remote.requests.Session', Client)
    handler = HandlerTTSRemote()
    handler.config = RemoteTTSConfig()
    context = RemoteTTSContext('test')
    context.request_id = 'old'
    control = threading.Thread(target=handler._cancel_worker, args=(context,))
    control.start()
    manager = SignalManager(None)
    manager.register_listener(lambda signal: handler.on_signal(context, signal))
    manager.register_listener(lambda signal: next_listener.set())
    manager.init()
    try:
        begin = time.monotonic()
        manager.signal_queue.put(NS(type=ChatSignalType.INTERRUPT,
                                    source_type=ChatSignalSourceType.CLIENT, related_stream=None))
        assert next_listener.wait(.2)
        assert time.monotonic() - begin < .2
        assert entered.wait(1)
    finally:
        release.set()
        context.stop.set()
        context.cancel_event.set()
        control.join(3)
        manager.shutdown()
    assert not control.is_alive()


def test_late_cancel_closes_only_captured_response(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    closed = []
    class Client:
        trust_env = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, *args, **kwargs):
            entered.set()
            release.wait(2)
    monkeypatch.setattr('handlers.tts.cosyvoice3_remote.tts_handler_cosyvoice3_remote.requests.Session', Client)
    handler = HandlerTTSRemote()
    handler.config = RemoteTTSConfig()
    context = RemoteTTSContext('test')
    context.request_id = 'old'
    context.response = NS(close=lambda: closed.append('old'))
    handler._cancel(context)
    assert context.generation == 1
    control = threading.Thread(target=handler._cancel_worker, args=(context,))
    control.start()
    try:
        assert entered.wait(1)
        with context.state_lock:
            context.request_id = 'new'
            context.response = NS(close=lambda: closed.append('new'))
        release.set()
        deadline = time.monotonic() + 1
        while not closed and time.monotonic() < deadline:
            time.sleep(.01)
        assert closed == ['old']
    finally:
        context.stop.set()
        context.cancel_event.set()
        release.set()
        control.join(3)


def test_queue_media_duration_and_age_survive_get_clear():
    q = ThreadSafeAsyncQueue(lambda item: item)
    q.put_nowait(.04)
    q.put_nowait(.08)
    time.sleep(.01)
    assert q.snapshot()['seconds'] == .12
    assert q.snapshot()['oldest_age_ms'] >= 5
    assert q.get_nowait() == .04
    assert q.snapshot()['seconds'] == .08
    q.clear()
    assert q.snapshot()['seconds'] == 0
    assert q.snapshot()['peak_seconds'] == .12
    assert q.snapshot()['cleared_items'] == 1


def test_actual_delegate_measures_audio_and_video_seconds():
    from handlers.client.rtc_client.client_handler_rtc import RtcClientSessionDelegate
    from chat_engine.data_models.engine_channel_type import EngineChannelType
    delegate = RtcClientSessionDelegate()
    audio = NS(data=NS(get_main_data=lambda: np.zeros((1, 2400)),
                       get_main_definition_entry=lambda: NS(sample_rate=24000)))
    video = NS(data=NS(get_main_data=lambda: np.zeros((1, 2, 2, 3))))
    delegate.output_queues[EngineChannelType.AUDIO].put_nowait(audio)
    delegate.output_queues[EngineChannelType.VIDEO].put_nowait(video)
    assert delegate.media_metrics()['queues']['audio']['seconds'] == .1
    assert delegate.media_metrics()['queues']['video']['seconds'] == .04
    delegate.clear_data()
    assert all(q['seconds'] == 0 for q in delegate.media_metrics()['queues'].values())


def test_cancel_before_response_cannot_emit_old_audio_or_close_new_request(monkeypatch):
    import base64
    import json
    from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry
    posted, release, finished = threading.Event(), threading.Event(), threading.Event()
    cancels, emitted = [], []
    class Response:
        def __init__(self, request): self.request = request
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
        def close(self): pass
        def raise_for_status(self): pass
        def iter_lines(self, **kwargs):
            yield json.dumps(dict(request_id=self.request['request_id'], sample_rate=24000,
                                  channels=1, dtype='float32le',
                                  audio=base64.b64encode(np.array([.5], dtype='<f4').tobytes()).decode())).encode()
            yield json.dumps(dict(request_id=self.request['request_id'], done=True)).encode()
    class Client:
        trust_env = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            if '/cancel/' in url:
                cancels.append(url.rsplit('/', 1)[-1])
                return None
            if kwargs['json']['text'] == 'old':
                posted.set()
                release.wait(2)
            return Response(kwargs['json'])
    monkeypatch.setattr('handlers.tts.cosyvoice3_remote.tts_handler_cosyvoice3_remote.requests.Session', Client)
    handler = HandlerTTSRemote()
    handler.config = RemoteTTSConfig()
    handler.definition = DataBundleDefinition()
    handler.definition.add_entry(DataBundleEntry.create_audio_entry('audio', 1, 24000))
    handler.definition.lockdown()
    context = RemoteTTSContext('test')
    def submit(bundle, finish_stream=False):
        emitted.append((bundle.get_main_data().copy(), finish_stream))
        if finish_stream: finished.set()
    context.submit_data = submit
    handler.start_context(None, context)
    try:
        context.jobs.put((0, 'old', time.monotonic()))
        assert posted.wait(1)
        old_id = context.request_id
        handler._cancel(context)
        context.jobs.put((context.generation, 'new', time.monotonic()))
        context.jobs.put((context.generation, None, time.monotonic()))
        release.set()
        assert finished.wait(2)
        assert len(emitted) == 2
        assert emitted[0][0].size == 1 and emitted[0][0][0, 0] == .5
        assert emitted[1][1]
        assert all(request == old_id for request in cancels)
    finally:
        release.set()
        handler.destroy_context(context)
    assert not context.worker.is_alive() and not context.cancel_worker.is_alive()


def test_input_queue_duration_reports_truncation_without_copying_pcm():
    import queue
    from engine_utils.media_metrics import input_queue_metrics
    from chat_engine.data_models.chat_data_type import ChatDataType
    q = queue.Queue()
    audio = NS(type=ChatDataType.AVATAR_AUDIO, data=NS(
        get_main_data=lambda: np.zeros((1, 2400)),
        get_main_definition_entry=lambda: NS(sample_rate=24000)))
    q.put(audio)
    q.put(audio)
    result = input_queue_metrics(q, limit=1)
    assert result == dict(items=2, audio_seconds=.1, truncated=True)
