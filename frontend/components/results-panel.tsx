"use client"

import { Activity, Download, Loader2, ChevronDown, ChevronUp } from "lucide-react"
import { useMemo, useState } from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ProbabilityGauge } from "@/components/probability-gauge"
import { ConfidenceMeter } from "@/components/confidence-meter"
import { ROCPRCurves } from "@/components/roc-pr-curves"
import { CVMetricsBars } from "@/components/cv-metrics-bars"
import { CalibrationCurve } from "@/components/calibration-curve"
import { FeatureImportance } from "@/components/feature-importance"
import { EmbeddingVisualization } from "@/components/embedding-visualization"
import { DemographicsCharts } from "@/components/demographics-charts"
import { MelSpectrogram } from "@/components/mel-spectrogram"
import { exportAnalysisReport } from "@/lib/export-report"
import type { ModelInsights } from "@/types/insights"

interface ResultsPanelProps {
  hasResults: boolean
  isAnalyzing: boolean
  probability: number
  confidence: number
  riskBand?: string
  spectrogram?: number[][]
  modelInsights?: ModelInsights | null
  insightsLoading?: boolean
  insightsError?: string | null
  onReset: () => void
}

export function ResultsPanel({
  hasResults,
  isAnalyzing,
  probability,
  confidence,
  riskBand,
  spectrogram,
  modelInsights,
  insightsLoading,
  insightsError,
  onReset,
}: ResultsPanelProps) {
  const [showDetailedStats, setShowDetailedStats] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)
  const riskLabel = useMemo(() => riskBand?.toUpperCase(), [riskBand])
  const classification = useMemo(() => {
    if (!hasResults) return "Awaiting analysis"
    return probability >= 50 ? "Likely Parkinson's" : "Likely Healthy Control"
  }, [hasResults, probability])
  const classificationColor = probability >= 50 ? "text-rose-400" : "text-emerald-400"

  const handleExport = async () => {
    if (!hasResults || isAnalyzing || exporting) return
    try {
      setExportError(null)
      setExporting(true)
      await exportAnalysisReport({
        probability,
        confidence,
        riskBand,
        insights: modelInsights ?? undefined,
      })
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "Export failed. Please try again.")
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg sm:text-xl">
            <Activity className="h-5 w-5 text-accent shrink-0" />
            Analysis Results
          </CardTitle>
          <CardDescription className="text-sm">AI prediction and confidence metrics</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {isAnalyzing ? (
            <div className="flex flex-col items-center justify-center py-8 sm:py-12 text-center animate-in fade-in duration-300">
              <div className="p-4 bg-primary/10 rounded-full mb-4">
                <Loader2 className="h-10 w-10 sm:h-12 sm:w-12 text-primary animate-spin" />
              </div>
              <p className="text-foreground font-medium text-base mb-1">Analyzing voice patterns...</p>
              <p className="text-muted-foreground text-sm px-4">Processing acoustic features and AI prediction</p>
            </div>
          ) : !hasResults ? (
            <div className="flex flex-col items-center justify-center py-8 sm:py-12 text-center">
              <div className="p-4 bg-muted rounded-full mb-4">
                <Activity className="h-10 w-10 sm:h-12 sm:w-12 text-muted-foreground" />
              </div>
              <p className="text-muted-foreground text-sm px-4">Upload or record audio to see analysis results</p>
            </div>
          ) : (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
              <div className="flex items-center justify-center animate-in zoom-in duration-500 delay-150">
                <ProbabilityGauge value={probability} />
              </div>

              <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-500 delay-300">
                <div className="text-center space-y-1">
                  <p className="text-sm text-muted-foreground">Predicted Probability</p>
                  <p className="text-2xl sm:text-3xl font-bold text-foreground">{probability.toFixed(1)}%</p>
                  {riskLabel && (
                    <span className="inline-flex items-center rounded-full border border-border px-3 py-0.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      {riskLabel} risk band
                    </span>
                  )}
                </div>
                <div className="rounded-lg border border-border/60 bg-black/30 px-4 py-3 text-sm text-muted-foreground">
                  <p className="uppercase text-[10px] tracking-[0.35em] text-muted-foreground">Classification</p>
                  <p className={`text-xl font-semibold ${classificationColor}`}>{classification}</p>
                  <p className="text-xs text-muted-foreground/70">
                    Threshold is 50% calibrated probability. Requires clinical validation.
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Confidence Level</span>
                    <span className="font-medium text-foreground">{confidence.toFixed(0)}%</span>
                  </div>
                  <ConfidenceMeter value={confidence} />
                </div>

                <MelSpectrogram data={spectrogram} />
              </div>

              <div className="flex flex-col gap-3 animate-in fade-in duration-500 delay-500">
                <Button
                  variant="outline"
                  className="w-full bg-transparent"
                  size="lg"
                  onClick={() => setShowDetailedStats(!showDetailedStats)}
                >
                  {showDetailedStats ? (
                    <>
                      <ChevronUp className="h-4 w-4 mr-2 shrink-0" />
                      <span className="truncate">Hide Detailed Statistics</span>
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-4 w-4 mr-2 shrink-0" />
                      <span className="truncate">Show Detailed Statistics</span>
                    </>
                  )}
                </Button>

                <Button
                  variant="outline"
                  className="w-full bg-transparent"
                  size="lg"
                  disabled={!hasResults || isAnalyzing || exporting}
                  onClick={handleExport}
                >
                  {exporting ? (
                    <Loader2 className="h-4 w-4 mr-2 shrink-0 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4 mr-2 shrink-0" />
                  )}
                  <span className="truncate">{exporting ? "Preparing PDF…" : "Export Report (PDF)"}</span>
                </Button>
                {exportError && (
                  <p className="text-xs text-destructive text-center">{exportError}</p>
                )}

                <Button onClick={onReset} variant="ghost" className="w-full h-12 text-sm">
                  Analyze Another Sample
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {showDetailedStats && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {insightsLoading ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Model diagnostics</CardTitle>
                <CardDescription className="text-xs">Fetching training telemetry…</CardDescription>
              </CardHeader>
              <CardContent className="py-8 text-center text-muted-foreground">
                <Loader2 className="mx-auto mb-3 h-6 w-6 animate-spin text-primary" />
                Loading insights from the backend
              </CardContent>
            </Card>
          ) : insightsError ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base text-destructive">Diagnostics unavailable</CardTitle>
                <CardDescription className="text-xs text-destructive">
                  {insightsError || "Unable to load dashboard data."}
                </CardDescription>
              </CardHeader>
            </Card>
          ) : modelInsights ? (
            <>
              <ROCPRCurves rocCurves={modelInsights.roc_curves} prCurves={modelInsights.pr_curves} />
              <CVMetricsBars metrics={modelInsights.cv_metrics} />
              <CalibrationCurve
                calibration={modelInsights.calibration}
                histogram={modelInsights.probability_histogram}
              />
              <FeatureImportance data={modelInsights.feature_importance} />
              <EmbeddingVisualization points={modelInsights.embedding} />
              <DemographicsCharts demographics={modelInsights.demographics} />
            </>
          ) : null}
        </div>
      )}
    </div>
  )
}
