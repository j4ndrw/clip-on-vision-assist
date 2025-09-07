import TabBasedRouter from "./tab-based-router";

import {
  Bluetooth as BluetoothIcon,
  AutoFixHigh as LlmIcon,
  Settings as PeripheralIcon,
  Wifi as WifiIcon,
  CellTower as HotspotIcon,
  Science as TestbedIcon,
  Storage as ComputeServerIcon,
} from "@mui/icons-material";

import Bluetooth from "./views/bluetooth";
import LlmConfiguration from "./views/llm-configuration";
import Peripherals from "./views/peripherals";
import Wifi from "./views/wifi";
import HotspotConfiguration from "./views/hotspot-configuration";
import Testbed from "./views/testbed";
import ComputeServerConfiguration from "./views/compute-server-configuration";

function App() {
  return (
    <TabBasedRouter
      title="Clip-On Vision Assist - Control Center"
      tabs={
        [
          {
            id: "wifi",
            Icon: () => <WifiIcon />,
            label: "Wi-Fi Configuration",
          },
          {
            id: "bluetooth",
            Icon: () => <BluetoothIcon />,
            label: "Bluetooth Settings",
          },
          {
            id: "peripherals",
            Icon: () => <PeripheralIcon />,
            label: "Peripheral Settings",
          },
          {
            id: "llm-configuration",
            Icon: () => <LlmIcon />,
            label: "LLM Configuration",
          },
          {
            id: "hotspot-configuration",
            Icon: () => <HotspotIcon />,
            label: "Hotspot Configuration",
          },
          {
            id: "compute-server-configuration",
            Icon: () => <ComputeServerIcon />,
            label: "Compute Server Configuration",
          },
          {
            id: "testbed",
            Icon: () => <TestbedIcon />,
            label: "Testbed",
          },
        ] as const
      }
      views={{
        wifi: Wifi,
        bluetooth: Bluetooth,
        peripherals: Peripherals,
        "llm-configuration": LlmConfiguration,
        "hotspot-configuration": HotspotConfiguration,
        "compute-server-configuration": ComputeServerConfiguration,
        testbed: Testbed,
      }}
    />
  );
}

export default App;
