import z from "zod";

export const requestSchema = z.object({
  cameraConfig: z.object({
    numFramesToCapture: z.number().gt(0),
    fps: z.number().gt(0),
    waitForNextBatchFactor: z.number().gt(0),
  }),
});
