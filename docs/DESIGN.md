# Design Specification — Browser RL Playground

## Goal

Train a PPO agent to balance an Inverted Double Pendulum (Phase 1) and eventually
walk a MuJoCo humanoid (Phase 2+), entirely in the browser with no backend.

---

## Environment: Inverted Double Pendulum

### Model

Use MuJoCo's built-in `inverted_double_pendulum.xml`.
- **Action space:** 1 continuous actuator (slider torque), clipped to [-1, 1]
- **Observation space:** 11-dimensional Float32Array
  - Cart position (1)
  - Sine/cosine of pole angles (4) — use sin/cos, not raw angles, for continuity
  - Cart velocity (1)
  - Angular velocities of both poles (2)
  - Constraint forces (3)

### Reward Function

```
alive_bonus  = 10.0  (per step, if not terminated)
height_cost  = -5.0 * (vertical_tip_position - target_height)²
control_cost = -0.1 * action²
velocity_cost = -0.01 * (v_cart² + ω₁² + ω₂²)

reward = alive_bonus + height_cost + control_cost + velocity_cost
```

### Termination

Episode ends when:
- Tip of the upper pole drops below 1.0m (fallen)
- Cart moves outside [-1.5, 1.5] range
- Episode reaches 1000 steps (timeout, not a failure — set `done=true` but
  do NOT bootstrap value to 0; instead bootstrap to V(s_final))

### Reset

On termination, reset MuJoCo state with small random noise:
- Joint positions: uniform [-0.01, 0.01]
- Joint velocities: uniform [-0.01, 0.01]

This avoids always starting from the exact same state.

---

## PPO Agent

### Network Architecture

Both networks use separate parameters (no shared backbone).

**Actor (Policy):**
```
obs (11) → Dense(64, tanh) → Dense(64, tanh) → action_mean (1, linear)
```
- `log_std`: a separate trainable `tf.variable`, shape [1], initialized to -0.5
- Action distribution: `Normal(action_mean, exp(log_std))`
- Sample action, then `tanh` squash to [-1, 1]
- Log-probability must include the tanh correction:
  `log_prob = normal_log_prob - Σ log(1 - tanh(a)² + 1e-6)`

**Critic (Value):**
```
obs (11) → Dense(64, tanh) → Dense(64, tanh) → value (1, linear)
```

**Weight initialization:**
- Dense layers: orthogonal initialization, gain = √2
- Output layers: orthogonal initialization, gain = 0.01 (actor), 1.0 (critic)
- This matters for stable early training.

### Hyperparameters

| Parameter            | Value   | Notes                                    |
|----------------------|---------|------------------------------------------|
| `gamma` (γ)          | 0.99    | Discount factor                          |
| `gae_lambda` (λ)     | 0.95    | GAE smoothing                            |
| `clip_epsilon` (ε)   | 0.2     | PPO clipping range                       |
| `learning_rate`      | 3e-4    | Adam optimizer                           |
| `rollout_length`     | 2048    | Steps per rollout before update          |
| `num_epochs` (K)     | 10      | Gradient passes over rollout buffer      |
| `mini_batch_size`    | 64      | Samples per gradient step                |
| `entropy_coeff`      | 0.01    | Entropy bonus (encourages exploration)   |
| `vf_coeff`           | 0.5     | Value function loss weight               |
| `max_grad_norm`      | 0.5     | Gradient clipping (global norm)          |
| `target_kl`          | 0.015   | Early stop epoch if KL exceeds this      |

### Training Loop (per rollout)

```
1. COLLECT phase (2048 steps):
   for t = 0 to 2047:
     action, log_prob = actor.sample(obs)
     value = critic(obs)
     next_obs, reward, done = env.step(action)
     buffer.store(obs, action, log_prob, reward, value, done)
     obs = next_obs  (or reset if done)

2. ADVANTAGE phase (GAE):
   bootstrap V(s_final) for last state
   for t = 2047 down to 0:
     if done[t]:  delta = reward[t] - value[t]
     else:        delta = reward[t] + γ * value[t+1] - value[t]
     advantage[t] = delta + γ * λ * (1 - done[t]) * advantage[t+1]
   returns = advantages + values
   normalize advantages to mean=0, std=1

3. UPDATE phase (K epochs × mini-batches):
   shuffle buffer indices
   for each mini-batch of 64:
     new_log_prob = actor.log_prob(obs_batch, action_batch)
     ratio = exp(new_log_prob - old_log_prob_batch)

     // Clipped surrogate loss
     surr1 = ratio * advantage_batch
     surr2 = clip(ratio, 1-ε, 1+ε) * advantage_batch
     policy_loss = -mean(min(surr1, surr2))

     // Value loss (clipped)
     new_value = critic(obs_batch)
     value_loss = mean((new_value - return_batch)²)

     // Entropy bonus
     entropy = -mean(new_log_prob)  // approximate

     // Total loss
     loss = policy_loss + vf_coeff * value_loss - entropy_coeff * entropy

     // Gradient step
     optimizer.minimize(loss)
     clip global gradient norm to max_grad_norm

   // Early stopping
   approx_kl = mean((ratio - 1) - log(ratio))
   if approx_kl > target_kl: break epoch loop
```

---

## TensorFlow.js Memory Rules

These are non-negotiable. Violating them will crash the browser tab.

1. **Every tensor operation in the update phase must be inside `tf.tidy()`.**
   `tf.tidy()` automatically disposes intermediate tensors.

2. **Rollout buffer tensors must be manually disposed after each update.**
   Create them as `tf.tensor()` from typed arrays, use them, then call `.dispose()`.

3. **Never store `tf.Tensor` objects in long-lived arrays.**
   Store raw `Float32Array` in the rollout buffer. Only convert to tensors
   inside the update function.

4. **Check for leaks during development:**
   ```ts
   const before = tf.memory().numTensors;
   agent.update(buffer);
   const after = tf.memory().numTensors;
   console.assert(before === after, `Leaked ${after - before} tensors`);
   ```

5. **Optimizer state tensors persist** — this is expected. But they should only
   be created once (when the optimizer is first used).

---

## WebWorker Communication Protocol

### Message Types (mujoco-protocol.ts)

```ts
// Main thread → Worker
type WorkerRequest =
  | { type: 'init'; modelXML: string }
  | { type: 'step'; action: Float32Array }
  | { type: 'reset' }
  | { type: 'render'; canvas: OffscreenCanvas }  // sent once via transfer

// Worker → Main thread
type WorkerResponse =
  | { type: 'ready' }
  | { type: 'step-result'; obs: Float32Array; reward: number; done: boolean }
  | { type: 'reset-result'; obs: Float32Array }
  | { type: 'frame'; bitmap: ImageBitmap }  // transferred, not cloned
```

### Transfer Rules

- `Float32Array` buffers: use structured clone (fast for small arrays)
- `OffscreenCanvas`: transfer once on init via `postMessage(msg, [canvas])`
- `ImageBitmap`: transfer each frame via `postMessage(msg, [bitmap])`
- NEVER send raw `ArrayBuffer` that is still referenced — it becomes detached

---

## Rendering Pipeline

1. Worker calls `mjr_render()` to draw into the OffscreenCanvas
2. Worker creates `ImageBitmap` from the canvas
3. Worker transfers `ImageBitmap` to main thread
4. Main thread draws bitmap to a visible `<canvas>` via `drawImage()`

Alternative (simpler, slightly slower): Worker posts raw pixel data as
`ImageData` and main thread calls `putImageData()`.

---

## UI State

```ts
interface AppState {
  isTraining: boolean;
  usePretrainedWeights: boolean;
  simSpeed: 1 | 5 | 10;           // render every Nth frame
  episodeRewards: number[];        // history for the graph
  currentEpisodeReward: number;
  currentStep: number;
  totalEpisodes: number;
  fps: number;
}
```

### Controls

- **Train / Pause** button
- **Load Pre-trained Weights** toggle (disabled during training)
- **Speed** slider: 1x / 5x / 10x (controls frame skip, not sim dt)
- **Reset** button (resets agent weights to random)

### Reward Graph (Recharts)

- X-axis: Episode number
- Y-axis: Total episode reward
- Rolling average line (window=20) overlaid on raw data points
- Update after each episode completes

---

## Phase 2 Scope (Future)

- Swap model XML to `humanoid.xml` or `walker2d.xml`
- Expand observation/action dimensions accordingly
- Larger networks: Dense(256) → Dense(256)
- Longer rollouts: 4096 steps
- Add camera controls for 3D viewing (this is where Three.js/R3F may enter)
