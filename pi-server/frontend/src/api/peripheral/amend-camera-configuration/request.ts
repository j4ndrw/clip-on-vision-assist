import { createApiRequest, createApiRequestConfig } from "@/api/utils";
import { requestSchema } from "./request.schema";

const config = createApiRequestConfig({
  method: 'POST',
  endpoint: '/api/peripheral/camera/config',
  requestSchema
});
export const request = createApiRequest(config);
