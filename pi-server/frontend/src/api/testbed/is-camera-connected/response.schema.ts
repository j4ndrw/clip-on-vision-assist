import z from "zod";

export const responseSchema = z.object({
  isCameraConnected: z.boolean(),
});
