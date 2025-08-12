import z from "zod";

export const responseSchema = z.object({
  endpointSuggestions: z.array(z.url())
});
