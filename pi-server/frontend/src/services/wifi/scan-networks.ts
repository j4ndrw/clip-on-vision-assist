import { wifiApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const scanNetworks = async (options: {
  onValidationError: (
    error: z.ZodError<
      z.infer<(typeof wifiApi)["scanNetworks"]["responseSchema"]>
    >,
  ) => void;
  onApiError: (error: ApiError) => void;
  onSuccess: (
    data: z.infer<(typeof wifiApi)["scanNetworks"]["responseSchema"]>,
  ) => void;
}) => {
  const { data, error, apiError } = await wifiApi.scanNetworks.request();

  if (error) {
    options.onValidationError(error);
    return;
  }

  if (apiError) {
    options.onApiError(apiError);
    return;
  }

  options.onSuccess(data);
};
