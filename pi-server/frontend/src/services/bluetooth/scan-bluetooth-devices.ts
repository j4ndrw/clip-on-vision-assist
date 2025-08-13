import { bluetoothApi } from "@/api";
import { createQueryService } from "../utils";

export const scanBluetoothDevices = createQueryService(bluetoothApi.getBluetoothDevices)
