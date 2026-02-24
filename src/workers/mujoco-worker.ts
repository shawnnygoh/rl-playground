// WebWorker: loads MuJoCo WASM, steps simulation, renders frames.
// Runs in a separate thread; communicates via WorkerRequest/WorkerResponse.

import type { WorkerRequest, WorkerResponse } from './mujoco-protocol'

// TODO: import and initialise mujoco-js
// import load from 'mujoco-js'

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const req = event.data

  switch (req.type) {
    case 'init': {
      // TODO: initialise MuJoCo with modelXML
      const resp: WorkerResponse = { type: 'ready' }
      self.postMessage(resp)
      break
    }

    case 'step': {
      // TODO: apply action and step simulation
      const obs = new Float32Array(11)
      const resp: WorkerResponse = { type: 'step-result', obs, reward: 0, done: false }
      self.postMessage(resp)
      break
    }

    case 'reset': {
      // TODO: reset simulation with small random noise
      const obs = new Float32Array(11)
      const resp: WorkerResponse = { type: 'reset-result', obs }
      self.postMessage(resp)
      break
    }

    case 'render': {
      // TODO: store OffscreenCanvas reference for rendering
      break
    }

    default: {
      // Exhaustive check
      const _exhaustive: never = req
      console.error('Unknown worker request:', _exhaustive)
    }
  }
}
