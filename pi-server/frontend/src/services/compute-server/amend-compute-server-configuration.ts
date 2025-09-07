import { computeServerApi } from "@/api";
import { createMutationService } from "../utils";

export const amendComputeServerConfiguration = createMutationService(computeServerApi.amendComputeServerConfiguration);
