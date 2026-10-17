from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('name,target,width,reset', [
    ('h264_nvenc', 1350000, 640, False),
    ('h264_nvenc', 1200000, 640, False),
    ('h264_nvenc', 750000, 640, True),
    ('h264_nvenc', 1500000, 1280, True),
    ('libx264', 1000000, 640, True),
])
def test_encoder_preserves_small_nvenc_changes_but_applies_large_drops(monkeypatch, name, target, width, reset):
    from aiortc.codecs import h264
    from handlers.client.rtc_client import client_handler_rtc as rtc

    created = []

    class Codec:
        def __init__(self, name):
            self.name = name
            self.width = 640
            self.height = 480
            self.bit_rate = 1500000
            self.closed = False

        def close(self):
            self.closed = True

        def encode(self, frame):
            return []

    def create(name, mode):
        assert mode == 'w'
        codec = Codec(name)
        created.append(codec)
        return codec

    # Keep the installed PyAV frame enum; replace codec construction only.
    monkeypatch.setattr(rtc.av, 'CodecContext', NS(create=create))
    monkeypatch.setattr(rtc, '_actual_h264_encoder', None)
    old = Codec(name)
    encoder = NS(codec=old, target_bitrate=target, _preferred_encoder=name,
                 buffer_data=b'', buffer_pts=None)
    frame = NS(width=width, height=480, pict_type=None)
    assert list(h264.H264Encoder._encode_frame(encoder, frame, False)) == []
    assert bool(created) is reset
    assert old.closed is reset
    if reset:
        assert encoder.codec is created[0]
        assert encoder.codec.bit_rate == target
        assert encoder.codec.width == width
    else:
        assert encoder.codec is old
