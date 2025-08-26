import { peripheralApi } from "@/api";
import { createMutationService } from "../utils";

export const amendMicrophoneConfiguration = createMutationService(peripheralApi.amendMicrophoneConfiguration)
