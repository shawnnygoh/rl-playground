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
      break
    }

    case 'reset': {
      const obs = doReset()
      const resp: WorkerResponse = { type: 'reset-result', obs }
      self.postMessage(resp)
      break
    }

    case 'render': {
      // Rendering wired up in a later phase
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
