// RewardGraph: recharts line graph of episode rewards with rolling average.
// X-axis: episode number. Y-axis: total episode reward.
// Overlays a rolling average (window=20) on raw data points.

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

interface RewardGraphProps {
  episodeRewards: number[]
  rollingWindow?: number
}

function rollingAverage(data: number[], window: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < window - 1) return null
    const slice = data.slice(i - window + 1, i + 1)
    return slice.reduce((a, b) => a + b, 0) / window
  })
}

export function RewardGraph({ episodeRewards, rollingWindow = 20 }: RewardGraphProps) {
  const avgRewards = rollingAverage(episodeRewards, rollingWindow)

  const chartData = episodeRewards.map((reward, i) => ({
    episode: i + 1,
    reward,
    avg: avgRewards[i],
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
        <YAxis label={{ value: 'Reward', angle: -90, position: 'insideLeft' }} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="reward" dot={false} stroke="#8884d8" name="Episode Reward" />
        <Line type="monotone" dataKey="avg" dot={false} stroke="#82ca9d" name={`Rolling Avg (${rollingWindow})`} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  )
}
