import z from "zod";
import { responseSchema } from './response.schema';

export type GetLlmEndpointSuggestionsResponse = z.infer<typeof responseSchema>
