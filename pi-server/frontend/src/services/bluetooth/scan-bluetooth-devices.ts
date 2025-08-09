import { bluetoothApi } from "@/api";
import type { ApiError } from "@/api/types";
import type z from "zod";

export const scanBluetoothDevices = async (options: {
  onValidationError: (
    error: z.ZodError<
      z.infer<(typeof bluetoothApi)["getBluetoothDevices"]["responseSchema"]>
    >,
  ) => void;
  onApiError: (error: ApiError) => void;
  onSuccess: (
    data: z.infer<(typeof bluetoothApi)["getBluetoothDevices"]["responseSchema"]>,
  ) => void;
}) => {
  const { data, error, apiError } = await bluetoothApi.getBluetoothDevices.request();

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
