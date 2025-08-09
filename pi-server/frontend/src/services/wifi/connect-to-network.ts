import { wifiApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const connectToNetwork = async (options: {
  input: z.infer<typeof wifiApi['connectToNetwork']['requestSchema']>,
  onApiError: (error: ApiError) => void;
  onSuccess: () => void;
}) => {
  const { apiError } = await wifiApi.connectToNetwork.request(options.input);

  if (apiError) {
    options.onApiError(apiError);
    return;
  }

  options.onSuccess();
};
