// All PPO hyperparameters as a typed config object.
// See docs/DESIGN.md for full rationale.

export interface PPOConfig {
  gamma: number        // Discount factor
  gaeLambda: number    // GAE smoothing
  clipEpsilon: number  // PPO clipping range
  learningRate: number // Adam optimizer learning rate
  rolloutLength: number  // Steps per rollout before update
  numEpochs: number    // Gradient passes over rollout buffer
  miniBatchSize: number  // Samples per gradient step
  entropyCoeff: number // Entropy bonus coefficient
  vfCoeff: number      // Value function loss weight
  maxGradNorm: number  // Gradient clipping (global norm)
  targetKL: number     // Early stop epoch if KL exceeds this

  // Environment
  obsDim: number       // Observation dimension (11 for inverted double pendulum)
  actionDim: number    // Action dimension (1 for inverted double pendulum)
}

export const DEFAULT_PPO_CONFIG: PPOConfig = {
  gamma: 0.99,
  gaeLambda: 0.95,
  clipEpsilon: 0.2,
  learningRate: 3e-4,
  rolloutLength: 2048,
  numEpochs: 10,
  miniBatchSize: 64,
  entropyCoeff: 0.01,
  vfCoeff: 0.5,
  maxGradNorm: 0.5,
  targetKL: 0.015,

  obsDim: 11,
  actionDim: 1,
}
