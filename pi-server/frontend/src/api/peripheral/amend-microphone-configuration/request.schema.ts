import z from "zod";

export const requestSchema = z.object({
  microphoneConfig: z.object({
    audioCaptureConfig: z.object({
      secondsPerChunk: z.number().gt(0),
      maxChunks: z.number().gt(0),
    }),
    silenceDetectionConfig: z.object({
      minSilenceLenMs: z.number().gte(0),
      silenceThresholdDbfs: z.number(),
    }),
  }),
});
