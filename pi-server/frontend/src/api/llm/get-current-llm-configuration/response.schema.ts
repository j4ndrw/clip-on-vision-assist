import z from "zod";

export const responseSchema = z.object({
  llmConfig: z.object({
    model: z.string(),
    endpoint: z.string().nullish(),
    apiKey: z.string(),
  }),
});
