import type { GetCurrentLlmConfigurationResponse } from "./types";

export const mock: GetCurrentLlmConfigurationResponse = {
  llmConfig: {
    model: "llama3:3b",
    apiKey: "MY-API-KEY-23182137219",
    endpoint: "https://some.llm.backend"
  }
}
