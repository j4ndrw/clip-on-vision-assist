import { peripheralApi } from "@/api";
import { createMutationService } from "../utils";

export const amendCameraConfiguration = createMutationService(peripheralApi.amendCameraConfiguration);
