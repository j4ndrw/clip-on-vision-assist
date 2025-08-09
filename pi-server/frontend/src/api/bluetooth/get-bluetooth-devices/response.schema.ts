import z from "zod";

export const responseSchema = z.object({
  bluetoothDevices: z.array(z.object({
    name: z.string().nullish(),
    macAddress: z.string()
  }))
});
