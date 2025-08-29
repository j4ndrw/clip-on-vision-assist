import Container from "@mui/material/Container";
import { useCameraTestbed } from "@/hooks/use-camera-testbed";
import Alert from "@mui/material/Alert";
import Typography from "@mui/material/Typography";
import { CAMERA_FEED_SOURCE } from "@/api/standalone/testbed/camera-feed";
import { useAlertSnackbars } from "@/hooks/use-alert-snackbars";

function CameraConfiguration() {
  const alertSnackbars = useAlertSnackbars();
  const { isCameraConnected, markCameraAsDisconnected } = useCameraTestbed({
    alertSnackbars,
  });
  if (isCameraConnected === false) {
    return (
      <Alert variant="outlined" severity="warning">
        Your camera isn't connected!
      </Alert>
    );
  }

  if (isCameraConnected === undefined) {
    return (
      <Alert variant="outlined" severity="info">
        Your camera is connecting...
      </Alert>
    );
  }

  return (
    <Container
      sx={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        gap: "1rem",
      }}
    >
      <Typography variant="overline">Camera feed</Typography>
      <img
        style={{
          maxWidth: "100%",
        }}
        src={CAMERA_FEED_SOURCE}
        onError={markCameraAsDisconnected}
      />
    </Container>
  );
}

export default CameraConfiguration;
