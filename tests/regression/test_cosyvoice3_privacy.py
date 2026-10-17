from types import SimpleNamespace as NS

from loguru import logger

from handlers.tts.cosyvoice3_remote.tts_handler_cosyvoice3_remote import (
    HandlerTTSRemote, RemoteTTSConfig, RemoteTTSContext,
)


def test_queue_logging_does_not_include_speech_text():
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    speech_text = '这段私人对话不得写入日志'
    handler = HandlerTTSRemote()
    handler.config = RemoteTTSConfig()
    context = RemoteTTSContext('privacy-test')
    context.input_key = 'default'
    inputs = NS(stream_id=None, data=NS(get_main_data=lambda: speech_text), is_last_data=True)
    try:
        handler.handle(context, inputs, {})
    finally:
        logger.remove(sink)
    assert context.jobs.get_nowait()[1] == speech_text
    queued_messages = [message for message in messages if 'text queued' in message]
    assert queued_messages
    assert f'chars={len(speech_text)}' in queued_messages[0]
    assert speech_text not in ''.join(messages)
