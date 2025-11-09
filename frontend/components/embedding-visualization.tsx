"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Scatter, ScatterChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, ZAxis } from "recharts"

import type { EmbeddingPoint } from "@/types/insights"

interface EmbeddingVisualizationProps {
  points?: EmbeddingPoint[]
}

export function EmbeddingVisualization({ points }: EmbeddingVisualizationProps) {
  const hasData = points && points.length > 0
  const healthyData = hasData ? points.filter((d) => d.label === "Healthy Control") : []
  const pdData = hasData ? points.filter((d) => d.label !== "Healthy Control") : []
  const healthyColor = "hsl(219, 86%, 78%)"
  const pdColor = "hsl(219, 86%, 55%)"
  const axisColor = "hsl(215, 20%, 78%)"

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Embedding Visualization</CardTitle>
        <CardDescription className="text-xs">1,536-d embeddings projected to 2D, colored by diagnosis</CardDescription>
      </CardHeader>
      <CardContent>
        {hasData ? (
          <>
            <ResponsiveContainer width="100%" height={350}>
              <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Component 1"
                  tick={{ fontSize: 10, fill: axisColor }}
                  stroke={axisColor}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Component 2"
                  tick={{ fontSize: 10, fill: axisColor }}
                  stroke={axisColor}
                />
                <ZAxis type="number" dataKey="split" range={[60, 60]} />
                <Tooltip
                  formatter={(value, name, props) => {
                    if (name === "split") {
                      return props.payload?.split
                    }
                    return value
                  }}
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "6px",
                    fontSize: "11px",
                  }}
                  cursor={{ strokeDasharray: "3 3" }}
                />
                <Scatter name="Healthy Control" data={healthyData} fill={healthyColor} opacity={0.65} />
                <Scatter name="Parkinson's" data={pdData} fill={pdColor} opacity={0.8} />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="flex items-center justify-center gap-6 mt-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: healthyColor, opacity: 0.65 }} />
                <span className="text-muted-foreground">Healthy Control</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: pdColor, opacity: 0.8 }} />
                <span className="text-muted-foreground">Parkinson&apos;s Disease</span>
              </div>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Embedding projection will appear after training.</p>
        )}
      </CardContent>
    </Card>
  )
}
