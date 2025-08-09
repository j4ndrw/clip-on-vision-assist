import { peripheralApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const amendMicrophoneConfiguration = async (options: {
  input: z.infer<
    (typeof peripheralApi)["amendMicrophoneConfiguration"]["requestSchema"]
  >;
  onApiError: (error: ApiError) => void;
  onSuccess: () => void;
}) => {
  const { apiError } = await peripheralApi.amendMicrophoneConfiguration.request(
    options.input,
  );

  if (apiError) {
    options.onApiError(apiError);
    return;
  }

  options.onSuccess();
};
