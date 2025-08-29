import { hotspotApi } from "@/api";
import { createQueryService } from "../utils";

export const getCurrentHotspotConfiguration = createQueryService(hotspotApi.getCurrentHotspotConfiguration)
