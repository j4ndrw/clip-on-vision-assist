import z from "zod";

export const requestSchema = z.object({
  ssid: z.string(),
  password: z.string(),
})
