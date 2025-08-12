import { llmApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const amendLlmConfiguration = async (options: {
  input: z.infer<
    (typeof llmApi)["amendLlmConfiguration"]["requestSchema"]
  >;
  onApiError: (error: ApiError) => void;
  onSuccess: () => void;
}) => {
  const { apiError } = await llmApi.amendLlmConfiguration.request(
    options.input,
  );

  if (apiError) {
    options.onApiError(apiError);
    return;
  }

  options.onSuccess();
};
