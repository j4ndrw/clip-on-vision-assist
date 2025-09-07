import { computeServerApi } from "@/api";
import { createQueryService } from "../utils";

export const getCurrentComputeServerConfiguration = createQueryService(computeServerApi.getCurrentComputeServerConfiguration)
