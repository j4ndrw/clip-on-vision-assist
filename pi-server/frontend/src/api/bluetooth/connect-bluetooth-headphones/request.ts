import { createApiRequest, createApiRequestConfig } from "@/api/utils";
import { requestSchema } from "./request.schema";

const config = createApiRequestConfig({
  method: "POST",
  endpoint: "/api/bluetooth/headphones",
  requestSchema,
});
export const request = createApiRequest(config);
