"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, XAxis, YAxis, ResponsiveContainer, Cell } from "recharts"

import type { FeatureImportancePoint } from "@/types/insights"

interface FeatureImportanceProps {
  data?: FeatureImportancePoint[]
}

export function FeatureImportance({ data }: FeatureImportanceProps) {
  const displayData =
    data && data.length > 0
      ? [...data].sort((a, b) => a.importance - b.importance)
      : null
  const themeBlue = "hsl(219, 86%, 60%)"
  const axisColor = "hsl(215, 20%, 78%)"

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Feature Importance (XGBoost gain)</CardTitle>
        <CardDescription className="text-xs">
          {displayData ? "Top embedding dimensions contributing to the classifier" : "Run training to compute importances"}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {displayData ? (
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={displayData} layout="vertical" margin={{ left: 80 }}>
              <XAxis type="number" tick={{ fontSize: 10, fill: axisColor }} stroke={axisColor} />
              <YAxis
                dataKey="feature"
                type="category"
                tick={{ fontSize: 11, fill: axisColor }}
                stroke={axisColor}
                width={75}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {displayData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={themeBlue}
                    opacity={0.4 + (index / displayData.length) * 0.5}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-muted-foreground">Feature importances will appear after the model is trained.</p>
        )}
      </CardContent>
    </Card>
  )
}
