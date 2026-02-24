import { useEffect } from 'react'
import { INVERTED_DOUBLE_PENDULUM_XML } from './models/inverted_double_pendulum'
import type { WorkerRequest, WorkerResponse } from './workers/mujoco-protocol'

function fmtObs(arr: Float32Array): string {
  return `[${Array.from(arr).map(v => v.toFixed(3)).join(', ')}]`
}

export default function App() {
  useEffect(() => {
    const worker = new Worker(
      new URL('./workers/mujoco-worker.ts', import.meta.url),
      { type: 'module' },
    )

    let active = true
    let stepNum = 0

    const sendRandomStep = () => {
      if (!active) return
      const action = new Float32Array([Math.random() * 2 - 1])
      worker.postMessage({ type: 'step', action } satisfies WorkerRequest)
    }

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const resp = event.data
      switch (resp.type) {
        case 'ready':
          console.log('[App] Worker ready — sending reset')
          worker.postMessage({ type: 'reset' } satisfies WorkerRequest)
          break

        case 'reset-result':
          stepNum = 0
          console.log('[App] reset → obs:', fmtObs(resp.obs))
          sendRandomStep()
          break

        case 'step-result':
          stepNum++
          console.log(
            `[App] step ${stepNum}: reward=${resp.reward.toFixed(4)} done=${resp.done}`,
            '\n  obs:', fmtObs(resp.obs),
          )
          if (resp.done) {
            worker.postMessage({ type: 'reset' } satisfies WorkerRequest)
          } else {
            sendRandomStep()
          }
          break

        case 'frame':
          break // rendering not wired up yet

        default: {
          const _exhaustive: never = resp
          console.error('[App] unknown worker response:', _exhaustive)
        }
      }
    }

    worker.onerror = (e) => console.error('[App] worker error:', e)

    console.log('[App] Sending init with inverted_double_pendulum model...')
    worker.postMessage({
      type: 'init',
      modelXML: INVERTED_DOUBLE_PENDULUM_XML,
    } satisfies WorkerRequest)

    return () => {
      active = false
      worker.terminate()
    }
  }, [])

  return (
    <div style={{ fontFamily: 'monospace', padding: 16 }}>
      <h1>RL Playground</h1>
      <p>MuJoCo physics loop running — open DevTools console to inspect obs / reward / done</p>
      <p style={{ color: '#888' }}>
        obs[0] = cart x &nbsp;|&nbsp; obs[1..4] = sin/cos pole angles &nbsp;|&nbsp;
        obs[5..7] = velocities &nbsp;|&nbsp; obs[8..10] = constraint forces
      </p>
    </div>
  )
}
