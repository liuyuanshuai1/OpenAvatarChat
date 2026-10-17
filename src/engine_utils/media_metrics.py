"""Read-only queue snapshots; no audio copies or text content are collected."""
from chat_engine.data_models.engine_channel_type import EngineChannelType
from itertools import islice


def input_queue_metrics(data_queue, limit=256):
    if data_queue is None:
        return None
    with data_queue.mutex:
        total = len(data_queue.queue)
        items = list(islice(data_queue.queue, limit))
    seconds = 0.0
    for item in items:
        if getattr(getattr(item, 'type', None), 'channel_type', None) != EngineChannelType.AUDIO:
            continue
        bundle = getattr(item, 'data', None)
        if bundle is None:
            continue
        array = bundle.get_main_data()
        rate = bundle.get_main_definition_entry().sample_rate
        if array is not None and rate:
            seconds += array.shape[-1] / rate
    return dict(items=total, audio_seconds=round(seconds, 4), truncated=total > limit)
