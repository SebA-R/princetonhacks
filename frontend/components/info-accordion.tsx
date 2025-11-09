import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Card } from "@/components/ui/card"
import { CheckCircle2, Brain, AlertTriangle } from "lucide-react"

export function InfoAccordion() {
  const processingSteps = [
    "Noise Reduction",
    "Audio Normalization",
    "Feature Extraction",
    "Embedding Generation",
    "AI Inference",
  ]

  return (
    <Card className="p-4 sm:p-6">
      <Accordion type="single" collapsible className="w-full space-y-2">
        <AccordionItem value="processing" className="border-b-0">
          <AccordionTrigger className="text-base sm:text-lg font-semibold py-4 hover:no-underline">
            Processing Details
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3 pt-2 pb-2">
              {processingSteps.map((step, index) => (
                <div key={index} className="flex items-center gap-3 py-1">
                  <CheckCircle2 className="h-5 w-5 text-accent flex-shrink-0" />
                  <span className="text-foreground text-sm sm:text-base">{step}</span>
                </div>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="model" className="border-b-0">
          <AccordionTrigger className="text-base sm:text-lg font-semibold py-4 hover:no-underline">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              Model Information
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4 pt-2 pb-2">
              <div>
                <p className="font-medium text-foreground mb-2 text-sm sm:text-base">Architecture</p>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  WavLM + Wav2Vec2 Fusion with XGBoost Classifier
                </p>
              </div>
              <div>
                <p className="font-medium text-foreground mb-2 text-sm sm:text-base">Training Dataset</p>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Trained on clinical datasets following methods in Adnan et al. (2025). Uses voice biomarkers for early
                  detection of Parkinson&rsquo;s indicators.
                </p>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="disclaimer" className="border-b-0">
          <AccordionTrigger className="text-base sm:text-lg font-semibold py-4 hover:no-underline">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              Important Disclaimer
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3 pt-2 pb-2">
              <p className="text-sm text-muted-foreground leading-relaxed">
                This system is designed for research and educational purposes only. It is not intended to be used as a
                medical diagnostic tool.
              </p>
              <p className="text-sm text-muted-foreground leading-relaxed">
                <strong className="text-foreground">Always consult a healthcare professional</strong> for clinical
                evaluation, diagnosis, and treatment decisions. This AI model provides probabilistic predictions that
                should not replace professional medical judgment.
              </p>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </Card>
  )
}
