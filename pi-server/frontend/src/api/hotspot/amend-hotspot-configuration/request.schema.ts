import z from "zod";

export const requestSchema = z.object({
  hotspotConfig: z.object({ ssid: z.string().min(3), password: z.string().min(8) }),
});
