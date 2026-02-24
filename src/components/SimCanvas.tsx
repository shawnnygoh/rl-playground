// SimCanvas: receives OffscreenCanvas frames from the MuJoCo worker
// and renders them to a visible <canvas> element via drawImage().

import { useEffect, useRef } from 'react'

interface SimCanvasProps {
  width?: number
  height?: number
}

export function SimCanvas({ width = 640, height = 480 }: SimCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    // TODO: listen for 'frame' messages from the MuJoCo worker
    // and call ctx.drawImage(bitmap, 0, 0) on each frame
  }, [])

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ border: '1px solid #333' }}
    />
  )
}
