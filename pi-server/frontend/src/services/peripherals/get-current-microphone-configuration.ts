import { peripheralApi } from "@/api";
import { createQueryService } from "../utils";

export const getCurrentMicrophoneConfiguration = createQueryService(peripheralApi.getCurrentMicrophoneConfiguration)
