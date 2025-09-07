import z from "zod";

export const requestSchema = z.object({
  endpoint: z.url(),
  apiKey: z.string(),
});
