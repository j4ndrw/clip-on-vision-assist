import { llmApi } from "@/api";
import { createQueryService } from "../utils";

export const getAvailableLlms = createQueryService(llmApi.getAvailableLlms)
