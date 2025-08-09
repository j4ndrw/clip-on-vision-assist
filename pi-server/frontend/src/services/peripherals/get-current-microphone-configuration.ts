import { peripheralApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const getCurrentMicrophoneConfiguration = async (options: {
  onValidationError: (
    error: z.ZodError<
      z.infer<(typeof peripheralApi)["getCurrentMicrophoneConfiguration"]["responseSchema"]>
    >,
  ) => void;
  onApiError: (error: ApiError) => void;
  onSuccess: (
    data: z.infer<(typeof peripheralApi)["getCurrentMicrophoneConfiguration"]["responseSchema"]>,
  ) => void;
}) => {
  const { data, error, apiError } = await peripheralApi.getCurrentMicrophoneConfiguration.request();

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
