import z from "zod";

export const responseSchema = z.object({
  wifiNetworks: z.array(
    z.object({
      ssid: z.string(),
      signalStrengthDbm: z.number(),
    }),
  ),
});
