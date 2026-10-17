const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const vm = require('node:vm')
const path = require('node:path')
const code = fs.readFileSync(path.join(__dirname, '../../deployment/production-2026-10-16/frontend-overlay/runtime-diagnostics.js'), 'utf8')
function environment(enabled = true) {
  let stats = []
  class Peer {
    connectionState = 'connected'
    events = {}
    addEventListener(name, fn) { this.events[name] = fn }
    createDataChannel() {
      return { events: {}, send() {}, addEventListener(name, fn) { this.events[name] = fn } }
    }
    async getStats() { return new Map(stats.map(s => [s.id, s])) }
  }
  const context = { window: { RTCPeerConnection: Peer }, location: { search: enabled ? '?diagnostics=1' : '' },
    URLSearchParams, performance: { now: () => 1 }, Date, Set, Map, WeakSet, Proxy, Reflect,
    navigator: { userAgent: 'test' }, document: { hidden: false, readyState: 'loading',
      addEventListener() {}, querySelectorAll: () => [] }, setInterval() {}, setTimeout() {} }
  vm.runInNewContext(code, context)
  return { window: context.window, setStats: value => { stats = value }, Peer }
}
test('diagnostics is opt in', () => {
  const env = environment(false)
  assert.equal(env.window.RTCPeerConnection, env.Peer)
  assert.equal(env.window.oacDiagnostics, undefined)
})
test('receive counters use deltas and exclude ICE and chat content', async () => {
  const env = environment()
  const pc = new env.window.RTCPeerConnection()
  assert.ok(pc instanceof env.Peer)
  const ch = pc.createDataChannel('chat')
  ch.send(JSON.stringify({ header: { name: 'SendHumanText', request_id: 'one' }, payload: { text: 'private secret' } }))
  env.setStats([{ id: 'video', type: 'inbound-rtp', kind: 'video', timestamp: 1000,
    framesDecoded: 10, jitterBufferDelay: .1, jitterBufferEmittedCount: 10, bytesReceived: 1000 },
    { id: 'candidate', type: 'remote-candidate', address: 'private address' }])
  await env.window.oacDiagnostics.sample()
  env.setStats([{ id: 'video', type: 'inbound-rtp', kind: 'video', timestamp: 3000,
    framesDecoded: 60, jitterBufferDelay: .3, jitterBufferEmittedCount: 60, bytesReceived: 3000 }])
  await env.window.oacDiagnostics.sample()
  const rows = env.window.oacDiagnostics.snapshot()
  const last = rows.filter(row => row.type === 'inbound').at(-1)
  assert.equal(last.decoded_fps, 25)
  assert.ok(Math.abs(last.jitter_buffer_ms - 4) < 1e-9)
  assert.equal(last.received_kbps, 8)
  assert.ok(!JSON.stringify(rows).includes('private'))
})
test('diagnostic storage is bounded and closed peers are released', async () => {
  const env = environment()
  const pc = new env.window.RTCPeerConnection()
  for (let i = 0; i < 3100; ++i) env.window.oacDiagnostics.mark()
  assert.equal(env.window.oacDiagnostics.snapshot().length, 3000)
  pc.connectionState = 'closed'
  await env.window.oacDiagnostics.sample()
})
