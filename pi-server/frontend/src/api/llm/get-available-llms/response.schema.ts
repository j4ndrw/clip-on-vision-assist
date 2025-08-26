import z from "zod";

export const responseSchema = z.object({
  llms: z.array(z.string())
})
