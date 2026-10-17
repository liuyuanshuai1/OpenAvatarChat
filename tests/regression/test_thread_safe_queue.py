import asyncio
import threading
import pytest
import time
from handlers.client.rtc_client.thread_safe_queue import ThreadSafeAsyncQueue


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




def test_cross_thread_queue_wakes_debug_loop_and_clear_discards_old_data():
    async def run():
        asyncio.get_running_loop().set_debug(True)
        q = ThreadSafeAsyncQueue()
        pending = asyncio.create_task(q.get())
        await asyncio.sleep(0)
        thread = threading.Thread(target=lambda: q.put_nowait('first'))
        thread.start()
        assert await asyncio.wait_for(pending, .5) == 'first'
        thread.join()
        # No scheduled callback contains a payload that can reinsert old data.
        thread = threading.Thread(target=lambda: (q.put_nowait('old'), q.clear()))
        thread.start()
        thread.join()
        await asyncio.sleep(.01)
        assert q.empty()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q.get(), .01)
        q.put_nowait('new')
        assert await q.get() == 'new'
    asyncio.run(run())
