import { testbedApi } from "@/api";
import { createQueryService } from "../utils";

export const isCameraConnected = createQueryService(testbedApi.isCameraConnected)
