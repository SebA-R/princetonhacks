import { Brain, AlertTriangle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

export function Header() {
  return (
    <header className="border-b border-border bg-card">
      <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-6 max-w-7xl">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 mb-4">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Brain className="h-6 w-6 sm:h-8 sm:w-8 text-primary" />
          </div>
          <div className="flex-1">
            <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold tracking-tight text-foreground">
              Parkinson&rsquo;s Voice Analysis
            </h1>
            <p className="text-sm sm:text-base text-muted-foreground mt-1 text-balance">
              AI-based early screening through voice biomarkers
            </p>
          </div>
        </div>
        <Alert className="bg-destructive/10 border-destructive/30">
          <AlertTriangle className="h-4 w-4 text-destructive shrink-0" />
          <AlertDescription className="text-destructive-foreground text-sm">
            For research and educational use only. Not a diagnostic tool.
          </AlertDescription>
        </Alert>
      </div>
    </header>
  )
}
