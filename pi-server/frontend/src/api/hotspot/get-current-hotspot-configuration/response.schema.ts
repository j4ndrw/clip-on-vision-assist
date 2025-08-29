import z from "zod";

export const responseSchema = z.object({
  hotspotConfig: z.object({ ssid: z.string() }),
});
