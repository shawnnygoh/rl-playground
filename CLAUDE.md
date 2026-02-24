# CLAUDE.md

## Project

Browser-based RL playground. A MuJoCo WASM simulation runs physics
in a WebWorker; a TF.js PPO agent trains entirely client-side.
Phase 1: Inverted Double Pendulum (balance). Phase 2+: walker.

## Stack

- React 18 + TypeScript + Vite
- mujoco-wasm (physics, runs in WebWorker)
- @tensorflow/tfjs with WebGL backend (PPO agent)
- recharts (live reward graph)

## Build & Run

```bash
npm install
npm run dev          # starts Vite dev server
npm run build        # production build
npm run lint         # eslint
npm run typecheck    # tsc --noEmit
```

## Dev Server Requirement

Vite must serve SharedArrayBuffer headers. This is configured in
vite.config.ts via a custom plugin. If headers are missing,
the MuJoCo worker will fail silently.

## Architecture

```
src/
  workers/
    mujoco-worker.ts     # WebWorker: loads WASM, steps simulation
    mujoco-protocol.ts   # typed message interface (WorkerRequest/WorkerResponse)
  agents/
    ppo.ts               # PPOAgent class (actor-critic, GAE, clipped update)
    ppo-config.ts        # all hyperparameters as a typed config object
  components/
    SimCanvas.tsx         # receives OffscreenCanvas frames, renders
    RewardGraph.tsx       # recharts line graph of mean episode reward
    Controls.tsx          # speed slider, train/load toggles
  models/                 # MuJoCo XML model files
  weights/                # pretrained weight JSON files
docs/
  DESIGN.md              # full RL spec: reward, PPO, architecture, protocols
```

## Key Constraints

- NO Three.js or react-three-fiber. Use MuJoCo's built-in mjr renderer.
- ALL tf.tidy() — every tensor operation must be wrapped. Memory leaks
  will crash the tab within minutes.
- Worker communication uses structured clone (Float32Array), never JSON
  serialization of typed arrays.
- Action/observation buffers are Float32Array throughout the pipeline.

## Design Spec

See `docs/DESIGN.md` for PPO hyperparameters, reward function design,
network architecture, training loop pseudocode, and worker protocol.

## Conventions

- Strict TypeScript (no `any` except WASM interop boundaries)
- Prefer named exports
- One component per file
- Worker message types must be exhaustively handled (switch + never)
