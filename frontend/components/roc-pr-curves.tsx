"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Line, LineChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Legend, Tooltip } from "recharts"

import type { CurveSeries, PrPoint, RocPoint } from "@/types/insights"

interface ROCPRCurvesProps {
  rocCurves?: {
    folds: CurveSeries<RocPoint>[]
    holdout: CurveSeries<RocPoint>
  }
  prCurves?: {
    folds: CurveSeries<PrPoint>[]
    holdout: CurveSeries<PrPoint>
  }
}

const themeBlue = "hsl(219, 86%, 60%)"
const themeBlueSoft = "hsl(219, 86%, 80%)"
const tooltipStyle = {
  background: "hsl(var(--card))",
  border: `1px solid ${themeBlue}`,
  borderRadius: "6px",
  fontSize: "11px",
}
const axisColor = "hsl(215, 20%, 78%)"

export function ROCPRCurves({ rocCurves, prCurves }: ROCPRCurvesProps) {
  const hasRoc = rocCurves && rocCurves.holdout.points.length > 0
  const hasPr = prCurves && prCurves.holdout.points.length > 0

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">ROC Curves</CardTitle>
          <CardDescription className="text-xs">Receiver Operating Characteristic per CV fold</CardDescription>
        </CardHeader>
        <CardContent>
          {hasRoc ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={rocCurves!.holdout.points}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis
                  type="number"
                  dataKey="fpr"
                  domain={[0, 1]}
                  label={{
                    value: "False Positive Rate",
                    position: "insideBottom",
                    offset: -5,
                    style: { fontSize: 11, fill: axisColor },
                  }}
                  tick={{ fontSize: 10, fill: axisColor }}
                  stroke={axisColor}
                />
                <YAxis
                  type="number"
                  domain={[0, 1]}
                  label={{
                    value: "True Positive Rate",
                    angle: -90,
                    position: "insideLeft",
                    style: { fontSize: 11, fill: axisColor },
                  }}
                  tick={{ fontSize: 10, fill: axisColor }}
                  stroke={axisColor}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: "11px", color: axisColor }} />
                {rocCurves!.folds.map((curve, index) => (
                  <Line
                    key={curve.label}
                    type="monotone"
                    data={curve.points}
                    dataKey="tpr"
                    name={curve.label}
                    stroke={themeBlueSoft}
                    strokeWidth={1.4}
                    opacity={0.4 + index * 0.1}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                <Line
                  type="monotone"
                  data={rocCurves!.holdout.points}
                  dataKey="tpr"
                  name={rocCurves!.holdout.label}
                  stroke={themeBlue}
                  strokeWidth={2.2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">Run training to populate ROC curves.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Precision-Recall Curves</CardTitle>
          <CardDescription className="text-xs">Precision vs Recall per CV fold</CardDescription>
        </CardHeader>
        <CardContent>
          {hasPr ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={prCurves!.holdout.points}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis
                  type="number"
                  dataKey="recall"
                  domain={[0, 1]}
                  label={{ value: "Recall", position: "insideBottom", offset: -5, style: { fontSize: 11, fill: axisColor } }}
                  tick={{ fontSize: 10, fill: axisColor }}
                  stroke={axisColor}
                />
                <YAxis
                  type="number"
                  domain={[0, 1]}
                  label={{ value: "Precision", angle: -90, position: "insideLeft", style: { fontSize: 11, fill: axisColor } }}
                  tick={{ fontSize: 10, fill: axisColor }}
                  stroke={axisColor}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: "11px", color: axisColor }} />
                {prCurves!.folds.map((curve, index) => (
                  <Line
                    key={curve.label}
                    type="monotone"
                    data={curve.points}
                    dataKey="precision"
                    name={curve.label}
                    stroke={themeBlueSoft}
                    strokeWidth={1.4}
                    opacity={0.4 + index * 0.1}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                <Line
                  type="monotone"
                  data={prCurves!.holdout.points}
                  dataKey="precision"
                  name={prCurves!.holdout.label}
                  stroke={themeBlue}
                  strokeWidth={2.2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">Run training to populate PR curves.</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
