"use client"

interface WaveformVisualizerProps {
  data: number[]
}

export function WaveformVisualizer({ data }: WaveformVisualizerProps) {
  const maxValue = Math.max(...data)
  const normalizedData = data.map((value) => (value / maxValue) * 100)

  return (
    <div className="w-full h-32 bg-secondary/50 rounded-lg border border-border p-4 overflow-hidden">
      <div className="flex items-center justify-center h-full gap-[2px]">
        {normalizedData.map((value, index) => (
          <div
            key={index}
            className="flex-1 bg-primary rounded-full transition-all duration-300 animate-in fade-in slide-in-from-bottom-2"
            style={{
              height: `${value}%`,
              minHeight: "4px",
              animationDelay: `${index * 5}ms`,
            }}
          />
        ))}
      </div>
    </div>
  )
}
