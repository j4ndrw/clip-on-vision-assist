import { peripheralApi } from "@/api";
import { createQueryService } from "../utils";

export const getCurrentCameraConfiguration = createQueryService(peripheralApi.getCurrentCameraConfiguration)
