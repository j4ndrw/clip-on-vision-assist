import z from "zod";
import { responseSchema } from './response.schema';

export type GetCurrentMicrophoneConfigurationResponse = z.infer<typeof responseSchema>
