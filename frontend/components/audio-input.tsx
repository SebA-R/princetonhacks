"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Mic, Upload, Loader2, Play, Square, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { InteractiveWaveform } from "@/components/interactive-waveform"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { predictVoice } from "@/lib/api"

interface AudioInputProps {
  onAnalysisStart: () => void
  onAnalysisComplete: (data: {
    probability: number
    confidence: number
    riskBand: string
    spectrogram: number[][]
    sampleRate: number
    durationSec: number
  }) => void
  onAnalysisError?: () => void
  onReset: () => void
  isAnalyzing: boolean
}

export function AudioInput({
  onAnalysisStart,
  onAnalysisComplete,
  onAnalysisError,
  onReset,
  isAnalyzing,
}: AudioInputProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
      }
    }
  }, [audioUrl])

  const startRecording = async () => {
    setError(null)

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError("Your browser doesn't support audio recording. Please try a different browser.")
        return
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" })
        const file = new File([blob], "recording.webm", { type: "audio/webm" })
        setAudioFile(file)
        const url = URL.createObjectURL(blob)
        setAudioUrl(url)
        stream.getTracks().forEach((track) => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (err: any) {
      console.error("[v0] Error accessing microphone:", err)

      if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setError("No microphone found. Please connect a microphone and try again.")
      } else if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setError("Microphone permission denied. Please enable microphone access in your browser settings.")
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        setError("Microphone is already in use by another application. Please close other apps and try again.")
      } else if (err.name === "OverconstrainedError") {
        setError("Could not start recording with the requested settings. Please try again.")
      } else {
        setError("Unable to access microphone. Please check your device and browser settings.")
      }
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    const file = e.target.files?.[0]
    if (file) {
      setAudioFile(file)
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
      }
      const url = URL.createObjectURL(file)
      setAudioUrl(url)
    }
  }

  const handleAnalyze = async () => {
    if (!audioFile) return

    onAnalysisStart()

    try {
      const prediction = await predictVoice(audioFile)
      onAnalysisComplete({
        probability: prediction.probability * 100,
        confidence: prediction.confidence,
        riskBand: prediction.risk_band,
        spectrogram: prediction.spectrogram,
        sampleRate: prediction.sample_rate,
        durationSec: prediction.duration_sec,
      })
    } catch (error) {
      console.error("[v0] Prediction failed:", error)
      setError(error instanceof Error ? error.message : "Unable to analyze this recording. Please try again.")
      onAnalysisError?.()
    }
  }

  const handleReset = () => {
    setAudioFile(null)
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl)
    }
    setAudioUrl(null)
    setError(null)
    onReset()
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg sm:text-xl">
          <Mic className="h-5 w-5 text-primary shrink-0" />
          Record or Upload Voice
        </CardTitle>
        <CardDescription className="text-sm">
          Record yourself saying &quot;aaahh&quot; for 3-10 seconds at a steady, comfortable pitch
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {error && (
          <Alert variant="destructive" className="animate-in fade-in slide-in-from-top-2 duration-300">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
          <h4 className="font-semibold text-sm text-foreground mb-2">Recording Instructions:</h4>
          <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
            <li>Find a quiet location with minimal background noise</li>
            <li>Hold your device at a comfortable distance</li>
            <li>Take a deep breath and say &quot;aaahh&quot; steadily</li>
            <li>Maintain the same pitch for 3-10 seconds</li>
            <li>Try to keep the volume consistent throughout</li>
          </ul>
        </div>

        <div className="space-y-4">
          <div className="flex flex-col gap-4">
            <Button
              onClick={isRecording ? stopRecording : startRecording}
              variant={isRecording ? "destructive" : "default"}
              className="w-full h-14 text-base font-medium"
              disabled={isAnalyzing}
            >
              {isRecording ? (
                <>
                  <Square className="h-5 w-5 mr-2 shrink-0" />
                  Stop Recording
                </>
              ) : (
                <>
                  <Mic className="h-5 w-5 mr-2 shrink-0" />
                  Start Recording
                </>
              )}
            </Button>
            <Button
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
              className="w-full h-14 text-base font-medium"
              disabled={isAnalyzing}
            >
              <Upload className="h-5 w-5 mr-2 shrink-0" />
              Upload Audio File
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/wav,audio/mp3,audio/webm"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>

          {audioFile && (
            <div className="p-4 bg-secondary rounded-lg border border-border animate-in fade-in slide-in-from-bottom-2 duration-300">
              <p className="text-sm text-muted-foreground mb-2">Selected file:</p>
              <p className="text-sm font-medium text-foreground break-all">{audioFile.name}</p>
            </div>
          )}
        </div>

        {audioUrl && (
          <div className="space-y-2 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <p className="text-sm font-medium text-foreground">Interactive Waveform</p>
            <InteractiveWaveform audioUrl={audioUrl} />
          </div>
        )}

        <div className="flex flex-col gap-3">
          <Button
            onClick={handleAnalyze}
            disabled={!audioFile || isAnalyzing}
            className="w-full h-14 text-base font-medium"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin shrink-0" />
                Analyzing Voice...
              </>
            ) : (
              <>
                <Play className="h-5 w-5 mr-2 shrink-0" />
                Analyze Voice
              </>
            )}
          </Button>

          {audioFile && !isAnalyzing && (
            <Button onClick={handleReset} variant="ghost" className="w-full h-12 text-sm">
              Clear & Start Over
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
