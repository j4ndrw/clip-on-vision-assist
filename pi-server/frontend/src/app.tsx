import TabBasedRouter from "./tab-based-router";

import {
  Bluetooth as BluetoothIcon,
  AutoFixHigh as LlmIcon,
  Settings as PeripheralIcon,
  Wifi as WifiIcon,
} from "@mui/icons-material";

import Bluetooth from "./views/bluetooth";
import LlmConfiguration from "./views/llm-configuration";
import Peripherals from "./views/peripherals";
import Wifi from "./views/wifi";

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
        ] as const
      }
      views={{
        wifi: Wifi,
        bluetooth: Bluetooth,
        peripherals: Peripherals,
        "llm-configuration": LlmConfiguration,
      }}
    />
  );
}

export default App;
