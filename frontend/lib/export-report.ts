import jsPDF from "jspdf"
import autoTable from "jspdf-autotable"

import type { ModelInsights } from "@/types/insights"

type ExportPayload = {
  probability: number
  confidence: number
  riskBand?: string
  insights?: ModelInsights | null
}

const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`

const humanizeKey = (key: string) =>
  key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())

export async function exportAnalysisReport({
  probability,
  confidence,
  riskBand,
  insights,
}: ExportPayload) {
  const doc = new jsPDF()
  const generatedAt = new Date()

  doc.setFontSize(18)
  doc.text("Parkinson's Voice Risk Report", 14, 20)

  doc.setFontSize(11)
  doc.setTextColor(80)
  doc.text(`Generated: ${generatedAt.toLocaleString()}`, 14, 28)

  doc.setFontSize(13)
  doc.setTextColor(20)
  doc.text("Inference Summary", 14, 38)

  doc.setFontSize(11)
  doc.setTextColor(40)
  const summaryLines = [
    `Calibrated Probability: ${probability.toFixed(1)}%`,
    `Confidence Level: ${confidence.toFixed(1)}%`,
    `Risk Band: ${riskBand ? riskBand.toUpperCase() : "N/A"}`,
  ]
  doc.text(summaryLines, 14, 46)

  if (insights?.holdout_metrics) {
    const holdoutRows = Object.entries(insights.holdout_metrics).map(([key, value]) => [
      humanizeKey(key),
      typeof value === "number" ? value.toFixed(3) : String(value),
    ])
    autoTable(doc, {
      head: [["Holdout Metric", "Value"]],
      body: holdoutRows,
      startY: 60,
      theme: "striped",
      headStyles: { fillColor: [38, 70, 183] },
    })
  }

  if (insights?.cv_metrics?.length) {
    const startY = doc.lastAutoTable ? doc.lastAutoTable.finalY + 10 : 80
    const rows = insights.cv_metrics.map((metric) => [metric.fold, (metric.accuracy * 100).toFixed(1) + "%", (metric.auroc * 100).toFixed(1) + "%"])
    autoTable(doc, {
      head: [["CV Fold", "Accuracy", "AUROC"]],
      body: rows,
      startY,
      theme: "striped",
      headStyles: { fillColor: [38, 70, 183] },
    })
  }

  const endY = doc.lastAutoTable ? doc.lastAutoTable.finalY + 12 : 110
  doc.setFontSize(10)
  doc.setTextColor(120)
  doc.text(
    "Notes: This report is for research and educational purposes only. Probabilities are calibrated; consult a clinician for any medical decision.",
    14,
    Math.min(endY, 280),
    { maxWidth: 180 },
  )

  const filename = `pd-voice-report-${generatedAt.toISOString().split("T")[0]}.pdf`
  doc.save(filename)
}

