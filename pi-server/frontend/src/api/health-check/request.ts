import { createApiRequest, createApiRequestConfig } from "@/api/utils";

const config = createApiRequestConfig({ method: "GET", endpoint: "/api/healthcheck", });
export const request = createApiRequest(config)
