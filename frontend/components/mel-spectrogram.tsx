"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface MelSpectrogramProps {
  data?: number[][]
}

export function MelSpectrogram({ data }: MelSpectrogramProps) {
  const hasData = data && data.length > 0 && data[0].length > 0
  const rows = hasData ? data.length : 0
  const cols = hasData ? data[0].length : 0

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Mel-Spectrogram Analysis</CardTitle>
        <CardDescription className="text-xs">Denoised waveform with annotated frequency regions</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="w-full h-64 bg-secondary/50 rounded-lg border border-border overflow-hidden relative flex items-center justify-center">
          {hasData ? (
            <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="mel-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="hsl(219, 90%, 70%)" stopOpacity="0.95" />
                  <stop offset="50%" stopColor="hsl(199, 90%, 65%)" stopOpacity="0.85" />
                  <stop offset="80%" stopColor="hsl(180, 85%, 60%)" stopOpacity="0.7" />
                  <stop offset="100%" stopColor="hsl(180, 50%, 45%)" stopOpacity="0.4" />
                </linearGradient>
              </defs>

              {data.map((row, rowIndex) =>
                row.map((value, colIndex) => {
                  const normalized = Math.max(0, Math.min(1, value))
                  const x = (colIndex / cols) * 100
                  const y = (rowIndex / rows) * 100
                  const width = 100 / cols
                  const height = 100 / rows
                  return (
                    <rect
                      key={`${rowIndex}-${colIndex}`}
                      x={`${x}%`}
                      y={`${y}%`}
                      width={`${width}%`}
                      height={`${height}%`}
                      fill="url(#mel-grad)"
                      opacity={normalized}
                    />
                  )
                }),
              )}

              <rect
                x="25%"
                y="35%"
                width="45%"
                height="30%"
                fill="none"
                stroke="hsl(var(--accent))"
                strokeWidth="2"
                strokeDasharray="4 2"
              />
              <text x="27%" y="33%" fill="hsl(195, 100%, 85%)" fontSize="11" fontWeight="600">
                Tremor Band (4-8 Hz)
              </text>
              <text x="50%" y="98%" fill="hsl(215, 25%, 85%)" fontSize="11" textAnchor="middle">
                Time (frames)
              </text>
              <text
                x="2%"
                y="50%"
                fill="hsl(215, 25%, 85%)"
                fontSize="11"
                transform="rotate(-90 2 50)"
                textAnchor="middle"
              >
                Mel bins
              </text>
            </svg>
          ) : (
            <p className="text-sm text-muted-foreground">Upload audio to visualize the mel-spectrogram</p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
