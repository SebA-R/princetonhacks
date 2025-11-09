"use client"

import { Progress } from "@/components/ui/progress"

interface ConfidenceMeterProps {
  value: number
}

export function ConfidenceMeter({ value }: ConfidenceMeterProps) {
  const getColorClass = (val: number) => {
    if (val < 50) return "[&>div]:bg-destructive"
    if (val < 80) return "[&>div]:bg-chart-3"
    return "[&>div]:bg-accent"
  }

  return (
    <div className="space-y-1">
      <Progress value={value} className={`h-3 ${getColorClass(value)}`} />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
    </div>
  )
}
