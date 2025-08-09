import { healthCheckApi } from "@/api";
import type { ApiError } from "@/api/types";

export const healthCheck = async (options: {
  onApiError: (error: ApiError) => void;
  onSuccess: () => void;
}) => {
  const { apiError } = await healthCheckApi.request();

  if (apiError) {
    options.onApiError(apiError);
    return;
  }

  options.onSuccess();
};
