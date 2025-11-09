"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { useWavesurfer } from "@wavesurfer/react"
import { Play, Pause, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"

interface InteractiveWaveformProps {
  audioUrl: string | null
}

export function InteractiveWaveform({ audioUrl }: InteractiveWaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const waveformWrapperRef = useRef<HTMLDivElement>(null)
  const [currentTime, setCurrentTime] = useState("0:00")
  const [duration, setDuration] = useState("0:00")
  const [progressPercent, setProgressPercent] = useState(0)
  const animationFrameRef = useRef<number | null>(null)
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }, [])

  const { wavesurfer, isReady, isPlaying } = useWavesurfer({
    container: containerRef,
    url: audioUrl || "",
    waveColor: "rgba(148, 163, 184, 0.3)",
    progressColor: "hsl(var(--primary))",
    cursorColor: "hsl(var(--accent))",
    cursorWidth: 2,
    barWidth: 3,
    barGap: 2,
    barRadius: 3,
    height: 128,
    normalize: true,
    interact: true,
    hideScrollbar: true,
    dragToSeek: true,
    backend: "WebAudio",
  })

  useEffect(() => {
    if (!wavesurfer || !isPlaying) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      return
    }

    const updateProgress = () => {
      if (wavesurfer && isPlaying) {
        const time = wavesurfer.getCurrentTime()
        const dur = wavesurfer.getDuration()

        setCurrentTime(formatTime(time))

        if (dur > 0) {
          setProgressPercent((time / dur) * 100)
        }

        animationFrameRef.current = requestAnimationFrame(updateProgress)
      }
    }

    animationFrameRef.current = requestAnimationFrame(updateProgress)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [wavesurfer, isPlaying, formatTime])

  useEffect(() => {
    if (!wavesurfer) return

    const handleReady = () => {
      setDuration(formatTime(wavesurfer.getDuration()))
    }

    const handleTimeUpdate = (time: number) => {
      if (!isPlaying) {
        setCurrentTime(formatTime(time))
        const dur = wavesurfer.getDuration()
        if (dur > 0) {
          setProgressPercent((time / dur) * 100)
        }
      }
    }

    const handleFinish = () => {
      setProgressPercent(0)
      setCurrentTime("0:00")
    }

    wavesurfer.on("ready", handleReady)
    wavesurfer.on("timeupdate", handleTimeUpdate)
    wavesurfer.on("finish", handleFinish)

    return () => {
      wavesurfer.un("ready", handleReady)
      wavesurfer.un("timeupdate", handleTimeUpdate)
      wavesurfer.un("finish", handleFinish)
    }
  }, [wavesurfer, isPlaying, formatTime])

  const handlePlayPause = () => {
    if (wavesurfer) {
      wavesurfer.playPause()
    }
  }

  const handleRestart = () => {
    if (wavesurfer) {
      wavesurfer.seekTo(0)
      wavesurfer.play()
    }
  }

  if (!audioUrl) {
    return (
      <div className="w-full h-48 bg-secondary/50 rounded-lg border border-border flex items-center justify-center">
        <p className="text-muted-foreground text-sm">Record or upload audio to see waveform</p>
      </div>
    )
  }

  return (
    <div className="w-full space-y-4">
      <div className="bg-gradient-to-br from-secondary/80 to-secondary/40 rounded-lg border border-border p-6 shadow-sm">
        <div ref={waveformWrapperRef} className="relative w-full">
          <div ref={containerRef} className="w-full" />

          {isReady && (
            <>
              {/* Horizontal progress bar at top */}
              <div className="absolute top-0 left-0 right-0 h-1 bg-secondary/30 rounded-full overflow-hidden">
                <div className="h-full bg-primary transition-none" style={{ width: `${progressPercent}%` }} />
              </div>

              {/* Vertical playhead bar */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-accent shadow-lg transition-none pointer-events-none"
                style={{ left: `${progressPercent}%` }}
              >
                {/* Playhead circle at top */}
                <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-3 h-3 bg-accent rounded-full shadow-md" />
              </div>
            </>
          )}
        </div>

        {!isReady && (
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground">Loading waveform...</p>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Button
            onClick={handlePlayPause}
            disabled={!isReady}
            size="sm"
            className="h-10 w-10 p-0 shadow-md hover:shadow-lg transition-shadow"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
          </Button>

          <Button
            onClick={handleRestart}
            disabled={!isReady}
            size="sm"
            variant="outline"
            className="h-10 w-10 p-0 bg-transparent"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>

        <div className="text-sm text-muted-foreground font-mono bg-secondary/50 px-3 py-1.5 rounded-md border border-border">
          <span className="text-foreground font-semibold">{currentTime}</span>
          <span className="mx-2">/</span>
          <span>{duration}</span>
        </div>
      </div>

      <p className="text-xs text-muted-foreground text-center">
        Click or drag on the waveform to seek • The colored bar shows playback progress
      </p>
    </div>
  )
}
