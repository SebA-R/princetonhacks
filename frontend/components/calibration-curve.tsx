"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Line, LineChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Bar, BarChart } from "recharts"

import type { HistogramBin } from "@/types/insights"

interface CalibrationCurveProps {
  calibration?: { predicted: number; actual: number }[]
  histogram?: HistogramBin[]
}

export function CalibrationCurve({ calibration, histogram }: CalibrationCurveProps) {
  const hasCalibration = calibration && calibration.length > 0
  const hasHistogram = histogram && histogram.length > 0
  const themeBlue = "hsl(219, 86%, 60%)"
  const themeBlueLight = "hsl(219, 86%, 78%)"
  const axisColor = "hsl(215, 20%, 78%)"

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Calibration Curve</CardTitle>
          <CardDescription className="text-xs">Reliability of predicted probabilities</CardDescription>
        </CardHeader>
        <CardContent>
          {hasCalibration ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={calibration}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis
                  type="number"
                  dataKey="predicted"
                  domain={[0, 1]}
                  label={{
                    value: "Predicted Probability",
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
                  label={{ value: "Observed Probability", angle: -90, position: "insideLeft", style: { fontSize: 11 } }}
                  tick={{ fontSize: 10, fill: axisColor }}
                  stroke={axisColor}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke={themeBlueLight}
                  strokeWidth={1.5}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Perfect Calibration"
                />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke={themeBlue}
                  strokeWidth={2.5}
                  dot={{ r: 4, fill: themeBlue }}
                  name="Observed"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">Calibration points will appear after training.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Probability Distribution</CardTitle>
          <CardDescription className="text-xs">Histogram of calibrated probabilities</CardDescription>
        </CardHeader>
        <CardContent>
          {hasHistogram ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis dataKey="bin" tick={{ fontSize: 9, fill: axisColor }} stroke={axisColor} interval={2} />
                <YAxis tick={{ fontSize: 10, fill: axisColor }} stroke={axisColor} />
                <Bar dataKey="count" fill={themeBlue} radius={[4, 4, 0, 0]} opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">Histogram will populate after training.</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
