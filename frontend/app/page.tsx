"use client"

import { useEffect, useState } from "react"

import { Header } from "@/components/header"
import { AudioInput } from "@/components/audio-input"
import { ResultsPanel } from "@/components/results-panel"
import { InfoAccordion } from "@/components/info-accordion"
import { fetchInsights } from "@/lib/api"
import type { ModelInsights } from "@/types/insights"

type AnalysisData = {
  probability: number
  confidence: number
  riskBand: string
  spectrogram: number[][]
  sampleRate: number
  durationSec: number
}

const emptyAnalysis: AnalysisData = {
  probability: 0,
  confidence: 0,
  riskBand: "low",
  spectrogram: [],
  sampleRate: 16_000,
  durationSec: 0,
}

export default function Home() {
  const [hasResults, setHasResults] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisData, setAnalysisData] = useState<AnalysisData>(emptyAnalysis)
  const [modelInsights, setModelInsights] = useState<ModelInsights | null>(null)
  const [insightsLoading, setInsightsLoading] = useState(true)
  const [insightsError, setInsightsError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true
    async function loadInsights() {
      setInsightsLoading(true)
      setInsightsError(null)
      try {
        const data = await fetchInsights()
        if (isMounted) {
          setModelInsights(data)
        }
      } catch (error) {
        if (isMounted) {
          setInsightsError(error instanceof Error ? error.message : "Unable to load insights.")
        }
      } finally {
        if (isMounted) {
          setInsightsLoading(false)
        }
      }
    }
    loadInsights()
    return () => {
      isMounted = false
    }
  }, [])

  const handleAnalysisComplete = (data: AnalysisData) => {
    setAnalysisData(data)
    setHasResults(true)
    setIsAnalyzing(false)
  }

  const handleAnalysisStart = () => {
    setIsAnalyzing(true)
    setHasResults(false)
  }

  const handleReset = () => {
    setHasResults(false)
    setIsAnalyzing(false)
    setAnalysisData(emptyAnalysis)
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 sm:px-6 py-6 sm:py-8 max-w-7xl">
        <div className="grid grid-cols-1 gap-6 sm:gap-8 mb-6 sm:mb-8">
          <AudioInput
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onAnalysisError={() => setIsAnalyzing(false)}
            onReset={handleReset}
            isAnalyzing={isAnalyzing}
          />
          <div>
            <ResultsPanel
              hasResults={hasResults}
              isAnalyzing={isAnalyzing}
              probability={analysisData.probability}
              confidence={analysisData.confidence}
              riskBand={analysisData.riskBand}
              spectrogram={analysisData.spectrogram}
              modelInsights={modelInsights}
              insightsLoading={insightsLoading}
              insightsError={insightsError}
              onReset={handleReset}
            />
          </div>
        </div>
        <InfoAccordion />
      </main>
    </div>
  )
}
