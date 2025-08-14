import { bluetoothApi } from "@/api";
import { createMutationService } from "../utils";

export const connectToBluetoothHeadphones = createMutationService(
  bluetoothApi.connectBluetoothHeadphones,
);
