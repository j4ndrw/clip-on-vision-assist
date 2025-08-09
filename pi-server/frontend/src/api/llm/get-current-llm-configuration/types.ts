import z from "zod";
import { responseSchema } from './response.schema';

export type GetCurrentLlmConfigurationResponse = z.infer<typeof responseSchema>
