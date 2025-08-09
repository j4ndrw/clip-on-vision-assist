import { bluetoothApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const connectToBluetoothHeadphones = async (options: {
  input: z.infer<
    (typeof bluetoothApi)["connectBluetoothHeadphones"]["requestSchema"]
  >;
  onApiError: (error: ApiError) => void;
  onSuccess: () => void;
}) => {
  const { apiError } = await bluetoothApi.connectBluetoothHeadphones.request(
    options.input,
  );

  if (apiError) {
    options.onApiError(apiError);
    return;
  }

  options.onSuccess();
};
