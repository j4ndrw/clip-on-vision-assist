import z from "zod";
import { responseSchema } from './response.schema';

export type GetCurrentCameraConfigurationResponse = z.infer<typeof responseSchema>
