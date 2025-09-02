import { createApiRequest, createApiRequestConfig } from "@/api/utils";
import { responseSchema } from "./response.schema";

const config = createApiRequestConfig({
  method: 'GET',
  endpoint: '/api/testbed/camera/check',
  responseSchema,
  allowInDevMode: true
})
export const request = createApiRequest(config);
