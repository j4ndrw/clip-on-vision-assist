import { mock } from './response.mock'
import { createApiRequest, createApiRequestConfig } from "@/api/utils"
import { responseSchema } from "./response.schema"

const config = createApiRequestConfig({
  method: 'GET',
  endpoint: "/api/llm/list",
  responseSchema: responseSchema,
  mock
})
export const request = createApiRequest(config)
