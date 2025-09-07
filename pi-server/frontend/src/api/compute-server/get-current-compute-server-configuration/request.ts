import { createApiRequest, createApiRequestConfig } from "@/api/utils";
import { responseSchema } from "./response.schema";
import { mock } from "./response.mock";

const config = createApiRequestConfig({
  method: 'GET',
  endpoint: '/api/compute-server/config',
  responseSchema,
  mock
})
export const request = createApiRequest(config);
