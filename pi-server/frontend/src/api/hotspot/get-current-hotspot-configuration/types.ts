import z from "zod";
import { responseSchema } from './response.schema';

export type GetCurrentHotspotConfigurationResponse = z.infer<typeof responseSchema>
