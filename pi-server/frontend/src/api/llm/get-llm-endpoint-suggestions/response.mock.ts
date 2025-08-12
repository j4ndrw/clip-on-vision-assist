import type { GetLlmEndpointSuggestionsResponse } from "./types";

export const mock: GetLlmEndpointSuggestionsResponse = {
  endpointSuggestions: [
    "https://some.llm.backend",
    "https://yet.another.llm.backend",
    "http://localhost:11434/v1",
    "https://localhost:11434/v1",
  ],
};
