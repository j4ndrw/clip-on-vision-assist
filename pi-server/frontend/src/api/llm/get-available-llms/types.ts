import z from "zod";
import { responseSchema } from './response.schema';

export type GetAvailableLlmsResponse = z.infer<typeof responseSchema>
