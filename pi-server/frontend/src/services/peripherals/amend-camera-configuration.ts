import { peripheralApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const amendCameraConfiguration = async (options: {
  input: z.infer<
    (typeof peripheralApi)["amendCameraConfiguration"]["requestSchema"]
  >;
  onApiError: (error: ApiError) => void;
  onSuccess: () => void;
}) => {
  const { apiError } = await peripheralApi.amendCameraConfiguration.request(
    options.input,
  );

  if (apiError) {
    options.onApiError(apiError);
    return;
  }

  options.onSuccess();
};
