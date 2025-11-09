"use client"

interface ProbabilityGaugeProps {
  value: number
}

export function ProbabilityGauge({ value }: ProbabilityGaugeProps) {
  const radius = 80
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value / 100) * circumference

  const getColor = (val: number) => {
    if (val < 30) return "stroke-accent"
    if (val < 70) return "stroke-chart-3"
    return "stroke-destructive"
  }

  return (
    <div className="relative w-48 h-48">
      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r={radius} className="stroke-secondary" strokeWidth="16" fill="none" />
        <circle
          cx="100"
          cy="100"
          r={radius}
          className={`${getColor(value)} transition-all duration-1000`}
          strokeWidth="16"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl font-bold text-foreground">{value.toFixed(1)}</div>
          <div className="text-xs text-muted-foreground mt-1">Probability</div>
        </div>
      </div>
    </div>
  )
}
