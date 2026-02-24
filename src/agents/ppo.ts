// PPOAgent: actor-critic agent with GAE and clipped PPO update.
// ALL tensor operations must be wrapped in tf.tidy() to prevent memory leaks.
// See docs/DESIGN.md for full algorithm spec.

import * as tf from '@tensorflow/tfjs'
import type { PPOConfig } from './ppo-config'

export interface RolloutBuffer {
  obs: Float32Array[]
  actions: Float32Array[]
  logProbs: Float32Array[]
  rewards: number[]
  values: number[]
  dones: boolean[]
}

export class PPOAgent {
  private readonly config: PPOConfig
  private actor!: tf.LayersModel
  private critic!: tf.LayersModel
  private optimizer!: tf.Optimizer

  constructor(config: PPOConfig) {
    this.config = config
    this.buildNetworks()
  }

  private buildNetworks(): void {
    // TODO: build actor and critic networks with orthogonal initialisation
    // Actor:  obs → Dense(64, tanh) → Dense(64, tanh) → action_mean (linear)
    // Critic: obs → Dense(64, tanh) → Dense(64, tanh) → value (linear)
    this.optimizer = tf.train.adam(this.config.learningRate)
  }

  /** Sample an action from the policy. Returns action and log-probability. */
  act(obs: Float32Array): { action: Float32Array; logProb: number; value: number } {
    return tf.tidy(() => {
      // TODO: implement actor forward pass with tanh squashing
      void obs
      return {
        action: new Float32Array(this.config.actionDim),
        logProb: 0,
        value: 0,
      }
    })
  }

  /** Run PPO update over the collected rollout buffer. */
  update(_buffer: RolloutBuffer): void {
    // TODO: implement GAE computation and PPO clipped update
    // All tensor ops must be inside tf.tidy()
  }

  getWeights(): tf.NamedTensorMap {
    // TODO: return serialisable weights for save/load
    return {}
  }

  setWeights(_weights: tf.NamedTensorMap): void {
    // TODO: load weights from a named tensor map
  }
}
