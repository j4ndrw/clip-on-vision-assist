import { create } from "zustand";
import type { ScanNetworksResponse } from "@/api/wifi/scan-networks/types";
import type { GetBluetoothDevicesResponse } from "@/api/bluetooth/get-bluetooth-devices/types";

export type WifiStore = {
  networks: ScanNetworksResponse["wifiNetworks"];
  setNetworks: (networks: ScanNetworksResponse["wifiNetworks"]) => void;
  selectedNetwork:
  | ScanNetworksResponse["wifiNetworks"][number]
  | undefined
  | null;
  selectNetwork: (
    network: ScanNetworksResponse["wifiNetworks"][number] | undefined | null,
  ) => void;
};

export type BluetoothStore = {
  devices: GetBluetoothDevicesResponse["bluetoothDevices"];
  setDevices: (
    devices: GetBluetoothDevicesResponse["bluetoothDevices"],
  ) => void;
};

export const useWifiStore = create<WifiStore>((set) => ({
  networks: [],
  setNetworks: (networks) =>
    set({ networks: networks.filter(({ ssid }) => !ssid.startsWith("\\x00")) }),
  selectedNetwork: undefined,
  selectNetwork: (network) => set({ selectedNetwork: network }),
}));

export const useBluetoothStore = create<BluetoothStore>((set) => ({
  devices: [],
  setDevices: (devices) => set({ devices }),
}));
