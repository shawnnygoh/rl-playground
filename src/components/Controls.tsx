// Controls: Train/Pause button, Load Pre-trained Weights toggle,
// Speed slider (1x/5x/10x frame skip), and Reset button.

interface ControlsProps {
  isTraining: boolean
  usePretrainedWeights: boolean
  simSpeed: 1 | 5 | 10
  onToggleTraining: () => void
  onTogglePretrainedWeights: () => void
  onChangeSpeed: (speed: 1 | 5 | 10) => void
  onReset: () => void
}

const SPEED_OPTIONS: Array<1 | 5 | 10> = [1, 5, 10]

export function Controls({
  isTraining,
  usePretrainedWeights,
  simSpeed,
  onToggleTraining,
  onTogglePretrainedWeights,
  onChangeSpeed,
  onReset,
}: ControlsProps) {
  return (
    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
      <button onClick={onToggleTraining}>
        {isTraining ? 'Pause' : 'Train'}
      </button>

      <label>
        <input
          type="checkbox"
          checked={usePretrainedWeights}
          onChange={onTogglePretrainedWeights}
          disabled={isTraining}
        />
        {' '}Load Pre-trained Weights
      </label>

      <label>
        Speed:{' '}
        {SPEED_OPTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onChangeSpeed(s)}
            style={{ fontWeight: simSpeed === s ? 'bold' : 'normal', marginLeft: '0.25rem' }}
          >
            {s}x
          </button>
        ))}
      </label>

      <button onClick={onReset}>Reset</button>
    </div>
  )
}
