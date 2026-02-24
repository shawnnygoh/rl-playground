/// <reference lib="webworker" />
// WebWorker: loads MuJoCo WASM, steps simulation, renders frames.
// Runs in a separate thread; communicates via WorkerRequest/WorkerResponse.

import loadMujoco from 'mujoco-js'
import type { MainModule, MjModel, MjData } from 'mujoco-js'
import type { WorkerRequest, WorkerResponse } from './mujoco-protocol'

// ─── Constants ────────────────────────────────────────────────────────────────

// Tip height when both poles are fully upright (two 0.6 m pole segments)
const TARGET_HEIGHT = 1.2
const MAX_STEPS = 1000
const CART_RANGE = 1.5

// ─── Module state ─────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MjModule = MainModule & { FS: any; MEMFS: any }

let mj: MjModule
let model: MjModel
let data: MjData
let stepCount = 0
let initialized = false
const pendingMessages: MessageEvent<WorkerRequest>[] = []

// Rendering state — canvas is transferred once via the 'render' message
let offscreenCanvas: OffscreenCanvas | null = null
let canvasCtx: OffscreenCanvasRenderingContext2D | null = null

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * mujoco-js exposes mjData arrays (qpos, qvel, ctrl, …) as Float64Array views
 * backed by WASM linear memory (via emscripten's memory_view binding).
 * Casting through unknown lets TypeScript accept the any-typed property.
 */
function asF64(ptr: unknown): Float64Array {
  return ptr as Float64Array
}

function getTipZ(): number {
  // site_xpos is Float64Array with 3 values per site; tip is the first site
  return asF64(data.site_xpos)[2]
}

function getObs(): Float32Array {
  const qpos = asF64(data.qpos)
  const qvel = asF64(data.qvel)
  const qfrc = asF64(data.qfrc_constraint)

  const obs = new Float32Array(11)
  obs[0] = qpos[0]             // cart x position
  obs[1] = Math.sin(qpos[1])  // sin(hinge1 angle)
  obs[2] = Math.cos(qpos[1])  // cos(hinge1 angle)
  obs[3] = Math.sin(qpos[2])  // sin(hinge2 angle)
  obs[4] = Math.cos(qpos[2])  // cos(hinge2 angle)
  obs[5] = qvel[0]             // cart x velocity
  obs[6] = qvel[1]             // hinge1 angular velocity
  obs[7] = qvel[2]             // hinge2 angular velocity
  obs[8] = qfrc[0]             // constraint force DOF 0
  obs[9] = qfrc[1]             // constraint force DOF 1
  obs[10] = qfrc[2]            // constraint force DOF 2
  return obs
}

function computeReward(action: number): number {
  const tipZ = getTipZ()
  const qvel = asF64(data.qvel)
  const aliveBonus = 10.0
  const heightCost = -5.0 * (tipZ - TARGET_HEIGHT) ** 2
  const controlCost = -0.1 * action ** 2
  const velocityCost = -0.01 * (qvel[0] ** 2 + qvel[1] ** 2 + qvel[2] ** 2)
  return aliveBonus + heightCost + controlCost + velocityCost
}

function checkDone(): boolean {
  const tipZ = getTipZ()
  const cartX = asF64(data.qpos)[0]
  return tipZ < 1.0 || Math.abs(cartX) > CART_RANGE || stepCount >= MAX_STEPS
}

// ─── Simulation ops ───────────────────────────────────────────────────────────

function doReset(): Float32Array {
  mj.mj_resetData(model, data)

  // Perturb joint positions and velocities with small uniform noise
  const qpos = asF64(data.qpos)
  const qvel = asF64(data.qvel)
  for (let i = 0; i < model.nq; i++) {
    qpos[i] += (Math.random() * 2 - 1) * 0.01
  }
  for (let i = 0; i < model.nv; i++) {
    qvel[i] += (Math.random() * 2 - 1) * 0.01
  }

  // Recompute derived quantities (xpos, site_xpos, …) from perturbed state
  mj.mj_forward(model, data)
  stepCount = 0
  return getObs()
}

function doStep(actionArr: Float32Array): { obs: Float32Array; reward: number; done: boolean } {
  const action = Math.max(-1, Math.min(1, actionArr[0]))

  // Apply control signal
  asF64(data.ctrl)[0] = action

  // Advance physics by one timestep (dt = 0.01 s from XML)
  mj.mj_step(model, data)
  stepCount++

  const reward = computeReward(action)
  const done = checkDone()
  const obs = getObs()
  return { obs, reward, done }
}

// ─── Rendering ────────────────────────────────────────────────────────────────

/**
 * Draw a 2D side-view (X–Z plane) of the pendulum onto offscreenCanvas, then
 * create an ImageBitmap and transfer it to the main thread.
 *
 * Bodies in this model (MuJoCo body indices):
 *   0  worldbody  (static origin)
 *   1  cart       (xpos[3..5])
 *   2  pole       (body origin coincides with cart)
 *   3  pole2      (body origin = top of pole1)
 * Site 0: tip     (top of pole2)
 *
 * xpos layout: flat Float64Array, 3 values per body.
 * site_xpos layout: flat Float64Array, 3 values per site.
 */
function renderFrame(): void {
  if (!canvasCtx || !offscreenCanvas) return

  const ctx = canvasCtx
  const W = offscreenCanvas.width
  const H = offscreenCanvas.height

  // ── World→canvas coordinate transform ────────────────────────────────────
  // z=0 (floor) maps to floorY; z increases upward so canvas y decreases.
  const scale = H * 0.60     // pixels per metre  (288 px/m at H=480)
  const floorY = H * 0.80    // y-pixel of world z=0
  const originX = W * 0.50   // x-pixel of world x=0

  const wx = (x: number) => originX + x * scale
  const wy = (z: number) => floorY  - z * scale

  // ── Clear ─────────────────────────────────────────────────────────────────
  ctx.fillStyle = '#0f0f1e'
  ctx.fillRect(0, 0, W, H)

  // ── Floor line ────────────────────────────────────────────────────────────
  ctx.strokeStyle = '#33334a'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(0, wy(0))
  ctx.lineTo(W, wy(0))
  ctx.stroke()

  // ── Rail (cart travel range) ───────────────────────────────────────────────
  ctx.strokeStyle = '#22223a'
  ctx.lineWidth = 4
  ctx.beginPath()
  ctx.moveTo(wx(-CART_RANGE), wy(0))
  ctx.lineTo(wx(CART_RANGE),  wy(0))
  ctx.stroke()

  // ── Read world positions ───────────────────────────────────────────────────
  const xpos    = asF64(data.xpos)      // 3 floats × nbody
  const sitePos = asF64(data.site_xpos) // 3 floats × nsite

  // Body 1: cart
  const cartX = xpos[1 * 3 + 0]
  const cartZ = xpos[1 * 3 + 2]

  // Body 3: pole2 origin = top of pole1 = bottom of pole2
  const midX = xpos[3 * 3 + 0]
  const midZ = xpos[3 * 3 + 2]

  // Site 0: tip = top of pole2
  const tipX = sitePos[0]
  const tipZ = sitePos[2]

  // ── Pole 1 (cart→mid) ─────────────────────────────────────────────────────
  ctx.strokeStyle = '#00cccc'
  ctx.lineWidth = 8
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(wx(cartX), wy(cartZ))
  ctx.lineTo(wx(midX),  wy(midZ))
  ctx.stroke()

  // ── Pole 2 (mid→tip) ──────────────────────────────────────────────────────
  ctx.strokeStyle = '#cc00cc'
  ctx.lineWidth = 8
  ctx.beginPath()
  ctx.moveTo(wx(midX), wy(midZ))
  ctx.lineTo(wx(tipX), wy(tipZ))
  ctx.stroke()

  // ── Cart body ─────────────────────────────────────────────────────────────
  const cartPxW = 40
  const cartPxH = 20
  ctx.fillStyle = '#d4a017'
  ctx.fillRect(
    wx(cartX) - cartPxW / 2,
    wy(cartZ) - cartPxH / 2,
    cartPxW,
    cartPxH,
  )

  // ── Joints ────────────────────────────────────────────────────────────────
  ctx.fillStyle = '#ffffff'
  ctx.beginPath()
  ctx.arc(wx(cartX), wy(cartZ), 5, 0, Math.PI * 2)
  ctx.fill()

  ctx.beginPath()
  ctx.arc(wx(midX), wy(midZ), 5, 0, Math.PI * 2)
  ctx.fill()

  // ── Tip marker ────────────────────────────────────────────────────────────
  ctx.fillStyle = '#ff4444'
  ctx.beginPath()
  ctx.arc(wx(tipX), wy(tipZ), 5, 0, Math.PI * 2)
  ctx.fill()

  // ── Transfer bitmap to main thread ────────────────────────────────────────
  createImageBitmap(offscreenCanvas).then(bitmap => {
    const resp: WorkerResponse = { type: 'frame', bitmap }
    self.postMessage(resp, [bitmap as unknown as Transferable])
  })
}

// ─── Initialisation ───────────────────────────────────────────────────────────

async function initMuJoCo(modelXML: string): Promise<void> {
  const module = await loadMujoco()
  mj = module as unknown as MjModule

  // Mount a virtual in-memory filesystem so loadFromXML can read the file
  mj.FS.mkdir('/working')
  mj.FS.mount(mj.MEMFS, { root: '.' }, '/working')
  mj.FS.writeFile('/working/model.xml', modelXML)

  model = mj.MjModel.loadFromXML('/working/model.xml')
  data = new mj.MjData(model)

  console.log(
    `[mujoco-worker] model loaded — nq=${model.nq} nv=${model.nv} nu=${model.nu} nsite=${model.nsite}`
  )
}

// ─── Message dispatch ─────────────────────────────────────────────────────────

function handleRequest(req: Exclude<WorkerRequest, { type: 'init' }>): void {
  switch (req.type) {
    case 'step': {
      const { obs, reward, done } = doStep(req.action)
      const resp: WorkerResponse = { type: 'step-result', obs, reward, done }
      self.postMessage(resp)
      renderFrame() // fire-and-forget; step-result already sent above
      break
    }

    case 'reset': {
      const obs = doReset()
      const resp: WorkerResponse = { type: 'reset-result', obs }
      self.postMessage(resp)
      renderFrame() // show the initial state after each reset
      break
    }

    case 'render': {
      offscreenCanvas = req.canvas
      canvasCtx = offscreenCanvas.getContext('2d')
      break
    }

    default: {
      const _exhaustive: never = req
      console.error('[mujoco-worker] unknown request type:', _exhaustive)
    }
  }
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const req = event.data

  if (req.type === 'init') {
    initMuJoCo(req.modelXML)
      .then(() => {
        initialized = true
        self.postMessage({ type: 'ready' } satisfies WorkerResponse)
        // Drain any messages that arrived before init finished
        const queued = pendingMessages.splice(0)
        for (const msg of queued) {
          handleRequest(msg.data as Exclude<WorkerRequest, { type: 'init' }>)
        }
      })
      .catch((err: unknown) => {
        console.error('[mujoco-worker] init failed:', err)
      })
    return
  }

  if (!initialized) {
    pendingMessages.push(event)
    return
  }

  handleRequest(req)
}
