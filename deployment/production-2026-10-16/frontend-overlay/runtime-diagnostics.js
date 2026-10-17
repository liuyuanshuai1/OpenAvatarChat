/* Opt-in playback diagnostics. No audio/video, chat text, or ICE addresses. */
(() => {
  if (!new URLSearchParams(location.search).has('diagnostics') || !window.RTCPeerConnection) return
  const NativePeer = window.RTCPeerConnection
  const peers = new Set()
  const previous = new Map()
  const rows = []
  const videos = new WeakSet()
  let peerId = 0
  let busy = false
  let status
  const record = (type, data = {}) => {
    rows.push({ type, wall: new Date().toISOString(), elapsed_ms: performance.now(), ...data })
    if (rows.length > 3000) rows.splice(0, rows.length - 3000)
  }
  const delta = (now, before, key) =>
    typeof now[key] === 'number' && typeof before?.[key] === 'number' ? now[key] - before[key] : null
  const watchChannel = (channel, id) => {
    channel.addEventListener('message', event => {
      try {
        const message = JSON.parse(event.data)
        if (message.type === 'media_metrics') record('server_queue', { peer: id, data: message.data })
        else if (message.header?.name === 'ChatSignal') {
          record('signal', { peer: id, signal: message.payload?.type, stream: message.payload?.stream_key })
        }
      } catch (_) { /* Non-JSON RTC keepalives are not diagnostic events. */ }
    })
    const send = channel.send.bind(channel)
    channel.send = data => {
      try {
        const message = JSON.parse(data)
        if (['SendHumanText', 'Interrupt'].includes(message.header?.name)) {
          record('outgoing', { peer: id, name: message.header.name, request_id: message.header.request_id })
        }
      } catch (_) { /* Never record arbitrary channel payloads. */ }
      return send(data)
    }
  }
  window.RTCPeerConnection = new Proxy(NativePeer, {
    construct(Target, args) {
      const pc = Reflect.construct(Target, args)
      const id = ++peerId
      peers.add({ pc, id })
      record('peer_created', { peer: id })
      const create = pc.createDataChannel.bind(pc)
      pc.createDataChannel = (...options) => {
        const channel = create(...options)
        watchChannel(channel, id)
        return channel
      }
      pc.addEventListener('datachannel', event => watchChannel(event.channel, id))
      pc.addEventListener('connectionstatechange', () => {
        record('connection', { peer: id, state: pc.connectionState })
      })
      return pc
    },
  })
  const watchVideo = video => {
    if (videos.has(video) || !video.requestVideoFrameCallback) return
    videos.add(video)
    let last
    video.addEventListener('pause', () => { last = undefined })
    video.addEventListener('play', () => { last = undefined })
    document.addEventListener('visibilitychange', () => { last = undefined })
    const frame = now => {
      if (last !== undefined && now - last > 150 && !document.hidden && !video.paused) {
        record('presentation_gap', { gap_ms: Math.round(now - last), ready_state: video.readyState })
      }
      last = document.hidden ? undefined : now
      if (video.isConnected) video.requestVideoFrameCallback(frame)
    }
    video.requestVideoFrameCallback(frame)
  }
  const sample = async () => {
    if (busy) return
    busy = true
    try {
      for (const entry of peers) {
        const { pc, id } = entry
        if (pc.connectionState === 'closed') {
          peers.delete(entry)
          for (const key of previous.keys()) if (key.startsWith(`${id}:`)) previous.delete(key)
          continue
        }
        const stats = await pc.getStats()
        stats.forEach(stat => {
          if (stat.type !== 'inbound-rtp') return
          const key = `${id}:${stat.id}`
          const old = previous.get(key)
          const seconds = old ? (stat.timestamp - old.timestamp) / 1000 : null
          const decoded = delta(stat, old, 'framesDecoded')
          const emitted = delta(stat, old, 'jitterBufferEmittedCount')
          const delay = delta(stat, old, 'jitterBufferDelay')
          record('inbound', {
            peer: id, kind: stat.kind || stat.mediaType, hidden: document.hidden,
            packets_lost_delta: delta(stat, old, 'packetsLost'),
            jitter_ms: typeof stat.jitter === 'number' ? stat.jitter * 1000 : null,
            jitter_buffer_ms: emitted > 0 && delay !== null ? delay / emitted * 1000 : null,
            decoded_fps: seconds > 0 && decoded !== null ? decoded / seconds : null,
            frames_dropped_delta: delta(stat, old, 'framesDropped'),
            freeze_count_delta: delta(stat, old, 'freezeCount'),
            freeze_seconds_delta: delta(stat, old, 'totalFreezesDuration'),
            decode_ms_per_frame: decoded > 0 && delta(stat, old, 'totalDecodeTime') !== null
              ? delta(stat, old, 'totalDecodeTime') / decoded * 1000 : null,
            concealed_samples_delta: delta(stat, old, 'concealedSamples'),
            silent_concealed_samples_delta: delta(stat, old, 'silentConcealedSamples'),
            received_kbps: seconds > 0 ? delta(stat, old, 'bytesReceived') * 8 / seconds / 1000 : null,
          })
          // Retain numeric counters only; do not retain candidate addresses.
          previous.set(key, Object.fromEntries(Object.entries(stat).filter(([, value]) => typeof value === 'number')))
        })
      }
      document.querySelectorAll('video').forEach(video => {
        watchVideo(video)
        if (video.getVideoPlaybackQuality) {
          const quality = video.getVideoPlaybackQuality()
          record('playback', { hidden: document.hidden, paused: video.paused,
            current_time: video.currentTime, ready_state: video.readyState,
            total_video_frames: quality.totalVideoFrames, dropped_video_frames: quality.droppedVideoFrames })
        }
      })
      if (status) status.textContent = `诊断：${rows.length}条 · 连接${peers.size}`
    } catch (error) {
      record('diagnostic_error', { error: error?.name || 'Error' })
    } finally { busy = false }
  }
  const mark = () => record('user_stutter_mark', { hidden: document.hidden })
  const exportData = () => {
    const data = { version: 1, user_agent: navigator.userAgent,
      note: 'Counters measure receive/decode/presentation; they do not prove fresh avatar content or speaker output.',
      rows: rows.slice() }
    const link = document.createElement('a')
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
    link.href = url
    link.download = `oac-playback-${Date.now()}.json`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
    return data
  }
  window.oacDiagnostics = { sample, mark, export: exportData, snapshot: () => rows.slice() }
  const panel = () => {
    const box = document.createElement('div')
    box.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:2147483647;background:#fff;color:#222;padding:10px;border:1px solid #aaa;border-radius:8px;font:13px sans-serif;box-shadow:0 2px 10px #0002'
    status = document.createElement('span')
    status.textContent = '播放诊断已开启'
    box.append(status)
    for (const [label, action] of [['标记卡顿', mark], ['导出记录', exportData]]) {
      const button = document.createElement('button')
      button.textContent = label
      button.style.cssText = 'margin-left:8px;padding:4px 8px;cursor:pointer'
      button.addEventListener('click', action)
      box.append(button)
    }
    document.body.append(box)
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', panel, { once: true })
  else panel()
  setInterval(sample, 2000)
})()
