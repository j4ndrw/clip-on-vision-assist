import { llmApi } from "@/api";
import { createQueryService } from "../utils";

export const getLlmEndpointSuggestions = createQueryService(
  llmApi.getLlmEndpointSuggestions,
);
