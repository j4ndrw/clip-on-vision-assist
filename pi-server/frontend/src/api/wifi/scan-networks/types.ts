import type z from "zod";
import type { responseSchema } from "./response.schema";

export type ScanNetworksResponse = z.infer<typeof responseSchema>
