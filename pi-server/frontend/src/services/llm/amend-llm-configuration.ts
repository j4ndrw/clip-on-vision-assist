import { llmApi } from "@/api";
import { createMutationService } from "../utils";

export const amendLlmConfiguration = createMutationService(llmApi.amendLlmConfiguration);
