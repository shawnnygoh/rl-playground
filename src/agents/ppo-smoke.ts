// Smoke test for PPOAgent.
//
// Call runSmokeTest() from the browser console or wire it to a button.
// It verifies:
//   1. sample() returns correctly shaped outputs.
//   2. sample() leaves no Tensor leaks.
//   3. update() leaves no Tensor leaks after optimizer state is warmed up.
//
// Per DESIGN.md §TensorFlow.js Memory Rules, rule 4:
//   const before = tf.memory().numTensors;
//   agent.update(buffer);
//   const after = tf.memory().numTensors;
//   console.assert(before === after, `Leaked ${after - before} tensors`);
//
// Note: the FIRST update() call initialises Adam moment tensors
// (2 tensors × numParams). This is expected and documented in the spec.
// The leak check is therefore performed on the SECOND call.

import * as tf from '@tensorflow/tfjs'
import { PPOAgent, type RolloutBuffer } from './ppo'
import { DEFAULT_PPO_CONFIG } from './ppo-config'

function randomF32(len: number): Float32Array {
  return Float32Array.from({ length: len }, () => (Math.random() - 0.5) * 0.2)
}

function assert(condition: boolean, msg: string): void {
  if (!condition) throw new Error(`[smoke] FAIL: ${msg}`)
}

export async function runSmokeTest(): Promise<void> {
  const cfg = DEFAULT_PPO_CONFIG
  console.log('[smoke] ── PPOAgent smoke test ──')
  console.log(`[smoke] TF.js backend: ${tf.getBackend()}`)

  // ------------------------------------------------------------------ //
  // 1. Construction
  // ------------------------------------------------------------------ //
  const tensorsAtStart = tf.memory().numTensors
  const agent = new PPOAgent(cfg)
  const tensorsAfterBuild = tf.memory().numTensors
  console.log(`[smoke] Tensors after build: ${tensorsAfterBuild} (delta +${tensorsAfterBuild - tensorsAtStart})`)

  // ------------------------------------------------------------------ //
  // 2. sample() correctness
  // ------------------------------------------------------------------ //
  const dummyObs = new Float32Array(cfg.obsDim) // all zeros
  const t0 = tf.memory().numTensors
  const s = agent.sample(dummyObs)
  const t1 = tf.memory().numTensors

  assert(s.action.length    === cfg.actionDim,  `action.length expected ${cfg.actionDim}, got ${s.action.length}`)
  assert(s.rawAction.length === cfg.actionDim,  `rawAction.length expected ${cfg.actionDim}, got ${s.rawAction.length}`)
  assert(typeof s.logProb   === 'number',        'logProb must be a number')
  assert(typeof s.value     === 'number',        'value must be a number')
  assert(s.action[0] >= -1 && s.action[0] <= 1, `squashed action ${s.action[0]} not in [-1, 1]`)
  assert(t0 === t1, `sample() leaked ${t1 - t0} tensors`)

  console.log(`[smoke] sample() OK — action=${s.action[0].toFixed(4)}, logProb=${s.logProb.toFixed(4)}, value=${s.value.toFixed(4)}`)

  // ------------------------------------------------------------------ //
  // 3. Build a small fake rollout buffer (T = 2 × miniBatchSize)
  // ------------------------------------------------------------------ //
  const T = 2 * cfg.miniBatchSize   // 128 steps — fast but covers >1 mini-batch
  const buffer: RolloutBuffer = {
    obs:      [],
    actions:  [],
    logProbs: [],
    rewards:  [],
    values:   [],
    dones:    [],
    lastObs:  randomF32(cfg.obsDim),
  }

  for (let t = 0; t < T; t++) {
    const obs = randomF32(cfg.obsDim)
    const { rawAction, logProb, value } = agent.sample(obs)
    buffer.obs.push(obs)
    buffer.actions.push(rawAction)          // raw (pre-tanh) — correct for update()
    buffer.logProbs.push(logProb)
    buffer.rewards.push(Math.random() * 10) // arbitrary positive reward
    buffer.values.push(value)
    buffer.dones.push(Math.random() < 0.05) // ~5% terminal steps
  }

  console.log(`[smoke] Rollout buffer built: ${T} steps`)

  // ------------------------------------------------------------------ //
  // 4. Warm-up update — initialises Adam moment tensors (expected growth)
  // ------------------------------------------------------------------ //
  const beforeWarmup = tf.memory().numTensors
  agent.update(buffer)
  const afterWarmup = tf.memory().numTensors
  console.log(`[smoke] Warm-up update: tensors ${beforeWarmup} → ${afterWarmup} (delta +${afterWarmup - beforeWarmup}, expected: Adam state init)`)

  // ------------------------------------------------------------------ //
  // 5. Leak test — second update must not change tensor count
  // ------------------------------------------------------------------ //
  const beforeUpdate = tf.memory().numTensors
  agent.update(buffer)
  const afterUpdate = tf.memory().numTensors
  const leaked = afterUpdate - beforeUpdate

  console.log(`[smoke] Leak test: tensors ${beforeUpdate} → ${afterUpdate} (delta ${leaked >= 0 ? '+' : ''}${leaked})`)
  assert(leaked === 0, `update() leaked ${leaked} tensors`)

  // ------------------------------------------------------------------ //
  // 6. getWeights / setWeights round-trip
  // ------------------------------------------------------------------ //
  const weights = agent.getWeights()
  const weightKeys = Object.keys(weights)
  assert(weightKeys.includes('logStd'),   'weights missing logStd')
  assert(weightKeys.some(k => k.startsWith('actor/')),  'weights missing actor keys')
  assert(weightKeys.some(k => k.startsWith('critic/')), 'weights missing critic keys')

  // Verify setWeights doesn't throw
  agent.setWeights(weights)
  console.log(`[smoke] getWeights/setWeights OK — ${weightKeys.length} weight tensors`)

  // Dispose the weight tensors returned by getWeights (caller owns them)
  Object.values(weights).forEach(w => w.dispose())

  // ------------------------------------------------------------------ //
  // Done
  // ------------------------------------------------------------------ //
  console.log('[smoke] ── ALL CHECKS PASSED ──')
  console.log(`[smoke] Final tensor count: ${tf.memory().numTensors}`)
}
