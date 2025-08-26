import type { ScanNetworksResponse } from "./types";

export const mock: ScanNetworksResponse = {
  wifiNetworks: [
    { ssid: "THE_BEST_WIFI_IN_TOWN", signalStrengthDbm: -50 },
    { ssid: "u_gon_get_hacked", signalStrengthDbm: -70 },
  ],
};
