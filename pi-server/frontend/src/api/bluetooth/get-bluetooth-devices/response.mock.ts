import type { GetBluetoothDevicesResponse } from "./types";

export const mock: GetBluetoothDevicesResponse = {
  bluetoothDevices: [
    { name: "Test device", macAddress: "DE:AD:BE:EF:00" },
    { name: null, macAddress: "CA:FE:BA:BE:00" }
  ]
}
