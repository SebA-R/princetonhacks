"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, XAxis, YAxis, ResponsiveContainer } from "recharts"

import type { DemographicsData } from "@/types/insights"

interface DemographicsChartsProps {
  demographics?: DemographicsData
}

export function DemographicsCharts({ demographics }: DemographicsChartsProps) {
  const hasAge = demographics && demographics.age_bins.length > 0
  const hasSex = demographics && demographics.sex_counts.length > 0
  const healthyColor = "hsl(219, 86%, 78%)"
  const pdColor = "hsl(219, 86%, 55%)"
  const axisColor = "hsl(215, 20%, 78%)"

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Age Distribution</CardTitle>
          <CardDescription className="text-xs">By diagnosis group (metadata)</CardDescription>
        </CardHeader>
        <CardContent>
          {hasAge ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={demographics!.age_bins}>
                <XAxis dataKey="range" tick={{ fontSize: 11, fill: axisColor }} stroke={axisColor} />
                <YAxis tick={{ fontSize: 10, fill: axisColor }} stroke={axisColor} />
              <Bar dataKey="hc" fill={healthyColor} name="Healthy Control" radius={[4, 4, 0, 0]} opacity={0.85} />
              <Bar dataKey="pd" fill={pdColor} name="Parkinson's" radius={[4, 4, 0, 0]} opacity={0.85} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">Age bins unavailable.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Sex Distribution</CardTitle>
          <CardDescription className="text-xs">Participant counts by sex and diagnosis</CardDescription>
        </CardHeader>
        <CardContent>
          {hasSex ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={demographics!.sex_counts}>
                <XAxis dataKey="sex" tick={{ fontSize: 11, fill: axisColor }} stroke={axisColor} />
                <YAxis tick={{ fontSize: 10, fill: axisColor }} stroke={axisColor} />
              <Bar dataKey="hc" fill={healthyColor} name="Healthy Control" radius={[4, 4, 0, 0]} opacity={0.85} />
              <Bar dataKey="pd" fill={pdColor} name="Parkinson's" radius={[4, 4, 0, 0]} opacity={0.85} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-muted-foreground">Sex distribution unavailable.</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
