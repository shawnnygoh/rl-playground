// SimCanvas: displays ImageBitmap frames transferred from the MuJoCo worker.
// Exposes SimCanvasHandle via forwardRef so the parent can push frames without
// going through React state (avoids re-renders on every physics step).

import { forwardRef, useImperativeHandle, useRef } from 'react'

export interface SimCanvasHandle {
  /** Draw a transferred bitmap onto the visible canvas, then close it. */
  drawBitmap(bitmap: ImageBitmap): void
}

interface SimCanvasProps {
  width?: number
  height?: number
}

export const SimCanvas = forwardRef<SimCanvasHandle, SimCanvasProps>(
  function SimCanvas({ width = 640, height = 480 }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useImperativeHandle(ref, () => ({
      drawBitmap(bitmap: ImageBitmap) {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
        bitmap.close()
      },
    }), [])

    return (
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ border: '1px solid #333', display: 'block' }}
      />
    )
  }
)
