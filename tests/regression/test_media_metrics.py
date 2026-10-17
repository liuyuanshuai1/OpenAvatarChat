from types import SimpleNamespace as NS
import numpy as np


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
