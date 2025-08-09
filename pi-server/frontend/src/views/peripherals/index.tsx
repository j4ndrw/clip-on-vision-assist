import TabBasedRouter from "@/tab-based-router";
import {
  Camera as CameraIcon,
  Mic as MicrophoneIcon,
} from "@mui/icons-material";
import CameraConfiguration from "./camera-configuration";
import MicrophoneConfiguration from "./microphone-configuration";

function Peripherals() {
  return (
    <TabBasedRouter
      title="Peripherals Configuration"
      tabs={
        [
          {
            id: "camera",
            Icon: () => <CameraIcon />,
            label: "Camera",
          },
          {
            id: "microphone",
            Icon: () => <MicrophoneIcon />,
            label: "Microphone",
          },
        ] as const
      }
      views={{
        camera: CameraConfiguration,
        microphone: MicrophoneConfiguration,
      }}
    />
  );
}

export default Peripherals;
