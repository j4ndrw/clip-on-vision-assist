import TabBasedRouter from "@/tab-based-router";
import {
  Camera as CameraIcon,
} from "@mui/icons-material";
import Camera from "./camera";

function Peripherals() {
  return (
    <TabBasedRouter
      tabs={
        [
          {
            id: "camera",
            Icon: () => <CameraIcon />,
            label: "Camera",
          },
        ] as const
      }
      views={{ camera: Camera }}
    />
  );
}

export default Peripherals;
