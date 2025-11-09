"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell, ReferenceLine } from "recharts"

import type { CvMetric } from "@/types/insights"

interface CVMetricsBarsProps {
  metrics?: CvMetric[]
}

export function CVMetricsBars({ metrics }: CVMetricsBarsProps) {
  const hasData = metrics && metrics.length > 0
  const meanAccuracy =
    hasData && metrics
      ? metrics.reduce((sum, m) => sum + m.accuracy, 0) / metrics.length
      : 0
  const stdAccuracy =
    hasData && metrics
      ? Math.sqrt(metrics.reduce((sum, m) => sum + Math.pow(m.accuracy - meanAccuracy, 2), 0) / metrics.length)
      : 0
  const meanAUROC =
    hasData && metrics
      ? metrics.reduce((sum, m) => sum + m.auroc, 0) / metrics.length
      : 0
  const stdAUROC =
    hasData && metrics
      ? Math.sqrt(metrics.reduce((sum, m) => sum + Math.pow(m.auroc - meanAUROC, 2), 0) / metrics.length)
      : 0
  const themeBlue = "hsl(219, 86%, 60%)"
  const themeBlueLight = "hsl(219, 86%, 78%)"
  const axisColor = "hsl(215, 20%, 78%)"

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/*
        Use a brighter thematic blue so the bars pop against the dark UI.
      */}
      {/*
        Colors
      */}
      {/*
        We'll compute within render to avoid recreating constants outside.
      */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">CV Accuracy per Fold</CardTitle>
          <CardDescription className="text-xs">
            {hasData
              ? `Mean: ${(meanAccuracy * 100).toFixed(1)}% ± ${(stdAccuracy * 100).toFixed(1)}%`
              : "Run training to populate CV metrics"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {hasData ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis dataKey="fold" tick={{ fontSize: 11, fill: axisColor }} stroke={axisColor} />
                <YAxis domain={[0.7, 1.0]} tick={{ fontSize: 10, fill: axisColor }} stroke={axisColor} />
                <ReferenceLine
                  y={meanAccuracy}
                  stroke={themeBlue}
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  label={{ value: "Mean", fontSize: 10, fill: themeBlue }}
                />
                <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                  {metrics!.map((_, index) => (
                    <Cell key={`cell-acc-${index}`} fill={themeBlue} opacity={0.9 - index * 0.05} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">No CV accuracy data yet.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">CV AUROC per Fold</CardTitle>
          <CardDescription className="text-xs">
            {hasData
              ? `Mean: ${(meanAUROC * 100).toFixed(1)}% ± ${(stdAUROC * 100).toFixed(1)}%`
              : "Run training to populate CV metrics"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {hasData ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis dataKey="fold" tick={{ fontSize: 11, fill: axisColor }} stroke={axisColor} />
                <YAxis domain={[0.7, 1.0]} tick={{ fontSize: 10, fill: axisColor }} stroke={axisColor} />
                <ReferenceLine
                  y={meanAUROC}
                  stroke={themeBlue}
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  label={{ value: "Mean", fontSize: 10, fill: themeBlue }}
                />
                <Bar dataKey="auroc" radius={[4, 4, 0, 0]}>
                  {metrics!.map((_, index) => (
                    <Cell key={`cell-auroc-${index}`} fill={themeBlueLight} opacity={0.9 - index * 0.05} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">No CV AUROC data yet.</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
