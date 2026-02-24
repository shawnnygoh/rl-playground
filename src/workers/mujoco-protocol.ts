// Typed message interface for WebWorker communication.
// Worker communication uses structured clone (Float32Array), never JSON
// serialization of typed arrays.

// Main thread → Worker
export type WorkerRequest =
  | { type: 'init'; modelXML: string }
  | { type: 'step'; action: Float32Array }
  | { type: 'reset' }
  | { type: 'render'; canvas: OffscreenCanvas } // sent once via transfer

// Worker → Main thread
export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'step-result'; obs: Float32Array; reward: number; done: boolean }
  | { type: 'reset-result'; obs: Float32Array }
  | { type: 'frame'; bitmap: ImageBitmap } // transferred, not cloned
