import { healthCheckApi } from "@/api";
import { createBareQueryService } from "./utils";

export const healthCheck = createBareQueryService(healthCheckApi)
