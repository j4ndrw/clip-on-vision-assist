import { hotspotApi } from "@/api";
import { createMutationService } from "../utils";

export const amendHotspotConfiguration = createMutationService(hotspotApi.amendHotspotConfiguration)
