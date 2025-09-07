import z from "zod";

export const responseSchema = z.object({
  computeServerConfig: z.object({
    endpoint: z.string().nullish(),
  }),
});
