// PPOAgent: actor-critic with GAE and clipped PPO update.
//
// Memory rules (DESIGN.md §TensorFlow.js Memory Rules):
//   - update(): wrap computeGradients inside tf.tidy so tidy disposes
//     forward-pass intermediates once the backward pass completes.
//   - sample(): wrap all ops in tf.tidy; call dataSync() inside, return
//     plain JS values so no Tensor escapes.
//   - Never store tf.Tensor in the RolloutBuffer — Float32Array only.
//   - Manually dispose all tensors returned by computeGradients /
//     clipByGlobalNorm after applyGradients.

import * as tf from '@tensorflow/tfjs'
import type { PPOConfig } from './ppo-config'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Raw rollout data collected from the environment. Store Float32Array only,
 *  never tf.Tensor, to avoid leaks across rollouts. */
export interface RolloutBuffer {
  obs: Float32Array[]       // [T] each is [obsDim]
  actions: Float32Array[]   // [T] each is [actionDim] — RAW pre-tanh actions
  logProbs: number[]        // [T] scalar log-prob with tanh correction
  rewards: number[]         // [T]
  values: number[]          // [T] critic estimates at collection time
  dones: boolean[]          // [T] true when episode terminated
  lastObs: Float32Array     // final observation, used to bootstrap V(s_T)
}

/** Return value of sample(). `action` goes to the environment; `rawAction`
 *  goes into the buffer so log-probs can be recomputed exactly in update(). */
export interface SampleResult {
  action: Float32Array    // tanh-squashed, in [-1, 1] — send to env
  rawAction: Float32Array // pre-tanh Gaussian sample — store in buffer
  logProb: number
  value: number
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

export class PPOAgent {
  private readonly cfg: PPOConfig
  private readonly actor: tf.Sequential
  private readonly critic: tf.Sequential
  // Separate trainable log-std; shape [actionDim], init to logStdInit
  private readonly logStd: tf.Variable
  private readonly optimizer: tf.Optimizer
  // Flat list of all trainable Variables for computeGradients
  private readonly allVars: tf.Variable[]

  constructor(config: PPOConfig) {
    this.cfg = config
    this.actor = this.buildActor()
    this.critic = this.buildCritic()
    this.logStd = tf.variable(
      tf.fill([config.actionDim], config.logStdInit),
      true,
      'logStd',
    )
    this.optimizer = tf.train.adam(config.learningRate)
    // Collect all trainable variables: actor weights + critic weights + logStd
    this.allVars = [
      ...this.actor.trainableWeights.map(w => w.val as tf.Variable),
      ...this.critic.trainableWeights.map(w => w.val as tf.Variable),
      this.logStd,
    ]
  }

  // -------------------------------------------------------------------------
  // Network construction
  // -------------------------------------------------------------------------

  private buildActor(): tf.Sequential {
    const s2 = Math.sqrt(2)
    const model = tf.sequential()
    model.add(tf.layers.dense({
      inputShape: [this.cfg.obsDim],
      units: 64,
      activation: 'tanh',
      kernelInitializer: tf.initializers.orthogonal({ gain: s2 }),
      biasInitializer: 'zeros',
    }))
    model.add(tf.layers.dense({
      units: 64,
      activation: 'tanh',
      kernelInitializer: tf.initializers.orthogonal({ gain: s2 }),
      biasInitializer: 'zeros',
    }))
    // Output layer: gain=0.01 keeps initial actions near zero
    model.add(tf.layers.dense({
      units: this.cfg.actionDim,
      activation: 'linear',
      kernelInitializer: tf.initializers.orthogonal({ gain: 0.01 }),
      biasInitializer: 'zeros',
    }))
    return model
  }

  private buildCritic(): tf.Sequential {
    const s2 = Math.sqrt(2)
    const model = tf.sequential()
    model.add(tf.layers.dense({
      inputShape: [this.cfg.obsDim],
      units: 64,
      activation: 'tanh',
      kernelInitializer: tf.initializers.orthogonal({ gain: s2 }),
      biasInitializer: 'zeros',
    }))
    model.add(tf.layers.dense({
      units: 64,
      activation: 'tanh',
      kernelInitializer: tf.initializers.orthogonal({ gain: s2 }),
      biasInitializer: 'zeros',
    }))
    // Output layer: gain=1.0 per spec
    model.add(tf.layers.dense({
      units: 1,
      activation: 'linear',
      kernelInitializer: tf.initializers.orthogonal({ gain: 1.0 }),
      biasInitializer: 'zeros',
    }))
    return model
  }

  // -------------------------------------------------------------------------
  // Inference
  // -------------------------------------------------------------------------

  /** Sample an action from the current policy.
   *
   *  All intermediate Tensors are created and disposed inside tf.tidy.
   *  dataSync() is called inside tidy so the return value contains only
   *  plain JS typed arrays / numbers — no Tensor escapes. */
  sample(obs: Float32Array): SampleResult {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return tf.tidy((): any => {
      const obsTensor = tf.tensor2d(obs, [1, this.cfg.obsDim])

      // --- Actor forward pass ---
      const actionMean = this.actor.predict(obsTensor) as tf.Tensor // [1, D]
      const std = tf.exp(this.logStd)                               // [D]

      // Sample from N(actionMean, std)
      const noise = tf.randomNormal([1, this.cfg.actionDim])
      const rawAction = actionMean.add(noise.mul(std))              // [1, D]

      // Tanh squash
      const squashedAction = tf.tanh(rawAction)                     // [1, D]

      // Log-prob = normal_log_prob − Σ log(1 − tanh(raw)² + ε)
      // normalLogProb:  −0.5·((raw−mean)/std)² − log(std) − 0.5·log(2π)
      const diff = rawAction.sub(actionMean).div(std)
      const normalLogProb = diff.square().mul(-0.5)
        .sub(this.logStd)
        .sub(0.5 * Math.log(2 * Math.PI))
      const tanhCorr = tf.log(
        tf.scalar(1).sub(squashedAction.square()).add(1e-6),
      )
      const logProb = normalLogProb.sub(tanhCorr).sum(1)            // [1]

      // --- Critic forward pass ---
      const valueTensor = this.critic.predict(obsTensor) as tf.Tensor // [1,1]

      return {
        action:    squashedAction.dataSync() as Float32Array,
        rawAction: rawAction.dataSync()      as Float32Array,
        logProb:   logProb.dataSync()[0],
        value:     valueTensor.dataSync()[0],
      } satisfies SampleResult
    }) as SampleResult
  }

  // -------------------------------------------------------------------------
  // PPO update
  // -------------------------------------------------------------------------

  /** Run one full PPO update over the collected rollout buffer.
   *
   *  Layout:
   *    1. Bootstrap V(s_T) via critic on lastObs.
   *    2. Compute GAE advantages and returns in pure JS.
   *    3. Normalize advantages.
   *    4. K epochs × mini-batches: gradient step with global-norm clipping.
   *    5. Per-epoch approxKL early-stopping check (full buffer, inside tidy). */
  update(buffer: RolloutBuffer): void {
    const { cfg } = this
    const T = buffer.rewards.length
    const {
      obsDim, actionDim, gamma, gaeLambda,
      numEpochs, miniBatchSize, clipEpsilon,
      entropyCoeff, vfCoeff, maxGradNorm, targetKL,
    } = cfg

    // ------------------------------------------------------------------
    // 1. Bootstrap V(s_T)
    // ------------------------------------------------------------------
    const bootstrapValue: number = tf.tidy((): number => {
      const t = tf.tensor2d(buffer.lastObs, [1, obsDim])
      return ((this.critic.predict(t) as tf.Tensor).dataSync() as Float32Array)[0]
    }) as unknown as number

    // ------------------------------------------------------------------
    // 2. GAE (pure JS — no Tensors needed)
    // ------------------------------------------------------------------
    const advantages = new Float32Array(T)
    const returns    = new Float32Array(T)

    let lastAdv = 0
    for (let t = T - 1; t >= 0; t--) {
      const nextVal = t === T - 1 ? bootstrapValue : buffer.values[t + 1]
      const delta = buffer.dones[t]
        ? buffer.rewards[t] - buffer.values[t]
        : buffer.rewards[t] + gamma * nextVal - buffer.values[t]
      // Reset advantage trace at episode boundaries
      lastAdv = buffer.dones[t] ? delta : delta + gamma * gaeLambda * lastAdv
      advantages[t] = lastAdv
    }

    // Returns: computed BEFORE normalising advantages
    for (let t = 0; t < T; t++) {
      returns[t] = advantages[t] + buffer.values[t]
    }

    // ------------------------------------------------------------------
    // 3. Normalise advantages  (mean=0, std=1)
    // ------------------------------------------------------------------
    let advMean = 0
    for (let t = 0; t < T; t++) advMean += advantages[t]
    advMean /= T

    let advVar = 0
    for (let t = 0; t < T; t++) advVar += (advantages[t] - advMean) ** 2
    advVar /= T
    const advStd = Math.sqrt(advVar + 1e-8)

    for (let t = 0; t < T; t++) advantages[t] = (advantages[t] - advMean) / advStd

    // ------------------------------------------------------------------
    // 4. Flatten buffer arrays for batch slicing
    // ------------------------------------------------------------------
    const obsFlat        = new Float32Array(T * obsDim)
    const actionsFlat    = new Float32Array(T * actionDim)
    const oldLogProbFlat = new Float32Array(T)

    for (let t = 0; t < T; t++) {
      obsFlat.set(buffer.obs[t],     t * obsDim)
      actionsFlat.set(buffer.actions[t], t * actionDim)
      oldLogProbFlat[t] = buffer.logProbs[t]
    }

    // ------------------------------------------------------------------
    // 5. PPO update loop
    // ------------------------------------------------------------------
    const indices = Array.from({ length: T }, (_, i) => i)
    let earlyStop = false

    for (let epoch = 0; epoch < numEpochs && !earlyStop; epoch++) {

      // Fisher-Yates shuffle
      for (let i = T - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[indices[i], indices[j]] = [indices[j], indices[i]]
      }

      // ----- mini-batch loop -----
      for (let start = 0; start < T; start += miniBatchSize) {
        const batchIdx = indices.slice(start, start + miniBatchSize)
        const B = batchIdx.length

        const batchObs         = new Float32Array(B * obsDim)
        const batchActions     = new Float32Array(B * actionDim)
        const batchOldLogProbs = new Float32Array(B)
        const batchAdvantages  = new Float32Array(B)
        const batchReturns     = new Float32Array(B)

        for (let i = 0; i < B; i++) {
          const t = batchIdx[i]
          batchObs.set(buffer.obs[t],     i * obsDim)
          batchActions.set(buffer.actions[t], i * actionDim)
          batchOldLogProbs[i] = buffer.logProbs[t]
          batchAdvantages[i]  = advantages[t]
          batchReturns[i]     = returns[t]
        }

        // Wrap computeGradients in tf.tidy.
        // Execution order:
        //   a) tf.tidy scope opens
        //   b) computeGradients runs forward pass (intermediates created)
        //   c) computeGradients runs backward pass (needs those intermediates)
        //   d) computeGradients returns {value, grads} — backward pass is done
        //   e) tf.tidy scope closes: disposes all intermediates from (b)
        //      except the tensors inside the returned {value, grads} object
        // This guarantees no forward-pass tensor leaks.
        const { value: lossVal, grads } = tf.tidy(() => {
          return this.optimizer.computeGradients(() => {

            const obsBatch     = tf.tensor2d(batchObs,     [B, obsDim])
            const actionsBatch = tf.tensor2d(batchActions, [B, actionDim])
            const oldLP        = tf.tensor1d(batchOldLogProbs)
            const advBatch     = tf.tensor1d(batchAdvantages)
            const retBatch     = tf.tensor1d(batchReturns)

            // --- new log-probs under current policy ---
            const actionMeans = this.actor.predict(obsBatch) as tf.Tensor
            const std         = tf.exp(this.logStd)
            const diff        = actionsBatch.sub(actionMeans).div(std)
            const normalLogP  = diff.square().mul(-0.5)
              .sub(this.logStd)
              .sub(0.5 * Math.log(2 * Math.PI))
            const squashed  = tf.tanh(actionsBatch)
            const tanhCorr  = tf.log(tf.scalar(1).sub(squashed.square()).add(1e-6))
            const newLogP   = normalLogP.sub(tanhCorr).sum(1)   // [B]

            // --- PPO ratio ---
            const logRatio = newLogP.sub(oldLP)
            const ratio    = tf.exp(logRatio)

            // --- clipped surrogate loss ---
            const surr1      = ratio.mul(advBatch)
            const surr2      = tf.clipByValue(ratio, 1 - clipEpsilon, 1 + clipEpsilon)
              .mul(advBatch)
            const policyLoss = tf.minimum(surr1, surr2).mean().neg()

            // --- value loss (MSE) ---
            const valuePreds = (this.critic.predict(obsBatch) as tf.Tensor).reshape([-1])
            const valueLoss  = valuePreds.sub(retBatch).square().mean()

            // --- entropy bonus (approximate: −E[log π]) ---
            const entropy = newLogP.neg().mean()

            return policyLoss
              .add(valueLoss.mul(vfCoeff))
              .sub(entropy.mul(entropyCoeff)) as tf.Scalar

          }, this.allVars)
        }) as { value: tf.Scalar; grads: tf.NamedTensorMap }

        // --- clip gradients by global norm, then apply ---
        const gradTensors = Object.values(grads)
        const [clippedGrads, globalNorm] = tf.clipByGlobalNorm(gradTensors, maxGradNorm)

        const clippedMap: tf.NamedTensorMap = {}
        Object.keys(grads).forEach((name, i) => {
          clippedMap[name] = clippedGrads[i]
        })
        this.optimizer.applyGradients(clippedMap)

        // --- dispose all owned tensors ---
        lossVal.dispose()
        gradTensors.forEach(g => g.dispose())
        clippedGrads.forEach(g => g.dispose())
        globalNorm.dispose()
      }

      // ----- per-epoch early stopping: approxKL on full buffer -----
      // Run inside tf.tidy; dataSync() converts to a number before tidy exits.
      const approxKL = tf.tidy((): number => {
        const obsBatch  = tf.tensor2d(obsFlat,        [T, obsDim])
        const actBatch  = tf.tensor2d(actionsFlat,    [T, actionDim])
        const oldLP     = tf.tensor1d(oldLogProbFlat)

        const actionMeans = this.actor.predict(obsBatch) as tf.Tensor
        const std         = tf.exp(this.logStd)
        const diff        = actBatch.sub(actionMeans).div(std)
        const normalLogP  = diff.square().mul(-0.5)
          .sub(this.logStd)
          .sub(0.5 * Math.log(2 * Math.PI))
        const squashed = tf.tanh(actBatch)
        const tanhCorr = tf.log(tf.scalar(1).sub(squashed.square()).add(1e-6))
        const newLogP  = normalLogP.sub(tanhCorr).sum(1)

        const logRatio = newLogP.sub(oldLP)
        const ratio    = tf.exp(logRatio)

        // approxKL = E[(r−1) − log r]  (Schulman's approximation)
        return ratio.sub(tf.scalar(1)).sub(logRatio).mean().dataSync()[0]
      }) as unknown as number

      if (approxKL > targetKL) {
        earlyStop = true
      }
    }
  }

  // -------------------------------------------------------------------------
  // Weight serialisation
  // -------------------------------------------------------------------------

  /** Returns all trainable weights keyed by stable string names.
   *  Suitable for JSON serialisation and storage in /weights. */
  getWeights(): tf.NamedTensorMap {
    const w: tf.NamedTensorMap = {}
    this.actor.getWeights().forEach((t, i)  => { w[`actor/${i}`]  = t })
    this.critic.getWeights().forEach((t, i) => { w[`critic/${i}`] = t })
    w['logStd'] = this.logStd
    return w
  }

  /** Restores weights from a map produced by getWeights(). */
  setWeights(weights: tf.NamedTensorMap): void {
    const actorCount  = this.actor.getWeights().length
    const criticCount = this.critic.getWeights().length
    this.actor.setWeights(
      Array.from({ length: actorCount },  (_, i) => weights[`actor/${i}`]),
    )
    this.critic.setWeights(
      Array.from({ length: criticCount }, (_, i) => weights[`critic/${i}`]),
    )
    this.logStd.assign(weights['logStd'])
  }
}
