import z from "zod";

export const requestSchema = z.object({ macAddress: z.string() });
