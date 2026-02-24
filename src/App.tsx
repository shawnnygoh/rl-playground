import { useEffect, useRef } from 'react'
import { INVERTED_DOUBLE_PENDULUM_XML } from './models/inverted_double_pendulum'
import type { WorkerRequest, WorkerResponse } from './workers/mujoco-protocol'
import { SimCanvas, type SimCanvasHandle } from './components/SimCanvas'

export default function App() {
  const simCanvasRef = useRef<SimCanvasHandle>(null)

  useEffect(() => {
    const worker = new Worker(
      new URL('./workers/mujoco-worker.ts', import.meta.url),
      { type: 'module' },
    )

    let active = true
    let stepNum = 0
    let episodeCount = 0
    let episodeReward = 0

    const sendRandomStep = () => {
      if (!active) return
      const action = new Float32Array([Math.random() * 2 - 1])
      worker.postMessage({ type: 'step', action } satisfies WorkerRequest)
    }

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const resp = event.data
      switch (resp.type) {
        case 'ready': {
          console.log('[App] Worker ready — sending canvas + reset')

          // Transfer a standalone OffscreenCanvas to the worker once.
          // The worker draws into it each step and posts ImageBitmaps back.
          const offscreen = new OffscreenCanvas(640, 480)
          worker.postMessage(
            { type: 'render', canvas: offscreen } satisfies WorkerRequest,
            [offscreen],
          )

          worker.postMessage({ type: 'reset' } satisfies WorkerRequest)
          break
        }

        case 'reset-result':
          stepNum = 0
          episodeReward = 0
          sendRandomStep()
          break

        case 'step-result':
          stepNum++
          episodeReward += resp.reward
          if (resp.done) {
            episodeCount++
            console.log(
              `[App] episode ${episodeCount} ended — steps=${stepNum} total_reward=${episodeReward.toFixed(2)}`
            )
            worker.postMessage({ type: 'reset' } satisfies WorkerRequest)
          } else {
            sendRandomStep()
          }
          break

        case 'frame':
          simCanvasRef.current?.drawBitmap(resp.bitmap)
          break

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
      <h1 style={{ marginTop: 0 }}>RL Playground</h1>
      <SimCanvas ref={simCanvasRef} width={640} height={480} />
      <p style={{ color: '#888', marginTop: 8, fontSize: 13 }}>
        Random actions — open DevTools console for episode stats
      </p>
    </div>
  )
}
