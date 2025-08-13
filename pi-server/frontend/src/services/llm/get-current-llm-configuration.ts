import { llmApi } from "@/api";
import { createQueryService } from "../utils";

export const getCurrentLlmConfiguration = createQueryService(llmApi.getCurrentLlmConfiguration)
