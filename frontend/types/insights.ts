export interface HistogramBin {
  bin: string
  count: number
}

export interface RocPoint {
  fpr: number
  tpr: number
}

export interface PrPoint {
  recall: number
  precision: number
}

export interface CurveSeries<T> {
  label: string
  points: T[]
}

export interface CvMetric {
  fold: string
  accuracy: number
  auroc: number
}

export interface FeatureImportancePoint {
  feature: string
  importance: number
}

export interface EmbeddingPoint {
  x: number
  y: number
  label: string
  split: string
}

export interface DemographicsData {
  age_bins: { range: string; hc: number; pd: number }[]
  sex_counts: { sex: string; hc: number; pd: number }[]
}

export interface ModelInsights {
  holdout_metrics: Record<string, number>
  cv_metrics: CvMetric[]
  roc_curves: {
    folds: CurveSeries<RocPoint>[]
    holdout: CurveSeries<RocPoint>
  }
  pr_curves: {
    folds: CurveSeries<PrPoint>[]
    holdout: CurveSeries<PrPoint>
  }
  calibration: { predicted: number; actual: number }[]
  probability_histogram: HistogramBin[]
  feature_importance: FeatureImportancePoint[]
  embedding: EmbeddingPoint[]
  demographics: DemographicsData
}

export interface PredictionResponse {
  probability: number
  risk_band: string
  confidence: number
  spectrogram: number[][]
  sample_rate: number
  duration_sec: number
}
