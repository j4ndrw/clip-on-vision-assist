import { llmApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const getAvailableLlms = async (options: {
  onValidationError: (
    error: z.ZodError<
      z.infer<(typeof llmApi)["getAvailableLlms"]["responseSchema"]>
    >,
  ) => void;
  onApiError: (error: ApiError) => void;
  onSuccess: (
    data: z.infer<(typeof llmApi)["getAvailableLlms"]["responseSchema"]>,
  ) => void;
}) => {
  const { data, error, apiError } = await llmApi.getAvailableLlms.request();

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
