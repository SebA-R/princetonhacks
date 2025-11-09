"use client"

export function SpectrogramDisplay() {
  return (
    <div className="w-full h-48 bg-secondary/50 rounded-lg border border-border overflow-hidden relative">
      <div className="absolute inset-0 opacity-60">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="spectro-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="hsl(264, 70%, 60%)" stopOpacity="0.8" />
              <stop offset="50%" stopColor="hsl(162, 70%, 60%)" stopOpacity="0.6" />
              <stop offset="100%" stopColor="hsl(264, 70%, 30%)" stopOpacity="0.3" />
            </linearGradient>
          </defs>
          {Array.from({ length: 50 }).map((_, i) => {
            const x = (i / 50) * 100
            const height = 30 + Math.sin(i * 0.5) * 20 + ((i % 7) - 3) * 3
            const clampedHeight = Math.max(10, Math.min(80, height))
            const opacity = 0.65 + ((i % 5) * 0.05)
            return (
              <rect
                key={i}
                x={`${x}%`}
                y={`${50 - clampedHeight / 2}%`}
                width="2%"
                height={`${clampedHeight}%`}
                fill="url(#spectro-grad)"
                opacity={Math.min(0.95, opacity)}
              />
            )
          })}
        </svg>
      </div>
      <div className="absolute inset-0 flex items-center justify-center">
        <p className="text-xs text-muted-foreground bg-card/80 px-3 py-1 rounded">Frequency analysis visualization</p>
      </div>
    </div>
  )
}
