import z from "zod";

export const requestSchema = z.object({
  model: z.string(),
  endpoint: z.string(),
  apiKey: z.string(),
});
