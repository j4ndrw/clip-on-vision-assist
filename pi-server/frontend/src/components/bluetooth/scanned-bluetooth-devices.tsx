import { useBluetoothStore } from "@/store";
import Chip from "@mui/material/Chip";
import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import { connectToBluetoothHeadphones } from "@/services/bluetooth/connect-to-bluetooth-headphones";
import { useState } from "react";
import { Bluetooth as BluetoothIcon } from "@mui/icons-material";
import CircularProgress from "@mui/material/CircularProgress";
import Snackbar from "@/design-system/snackbar";
import { useAlertSnackbars } from "@/hooks/use-alert-snackbars";

const ScannedBluetoothDevices: React.FC<{ preconnectFn: () => void }> = ({
  preconnectFn,
}) => {
  const {
    snackbarSuccessMessage,
    snackbarErrorMessage,
    setSnackbarSuccessMessage,
    setSnackbarErrorMessage,
  } = useAlertSnackbars();
  const { devices } = useBluetoothStore();

  const [selectedDevice, setSelectedDevice] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSnackbarClose = () => {
    setSelectedDevice("");
    setSnackbarSuccessMessage("");
    setSnackbarErrorMessage("");
    setLoading(false);
  };

  const handleConnectClick = (device: (typeof devices)[number]) => async () => {
    preconnectFn();
    setLoading(true);
    setSelectedDevice(device.macAddress);
    connectToBluetoothHeadphones({
      input: { macAddress: device.macAddress },
      onApiError: (error) => {
        setSelectedDevice("");
        setSnackbarSuccessMessage("");
        setSnackbarErrorMessage(error.message);
        setLoading(false);
      },
      onSuccess: () => {
        setSelectedDevice("");
        setSnackbarSuccessMessage(
          `Successfully connected to bluetooth device \`${device.name ?? device.macAddress}\``,
        );
        setSnackbarErrorMessage("");
        setLoading(false);
      },
    }).promise();
  };

  return (
    <>
      <Container
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "0.5rem",
        }}
      >
        {devices.length > 0 ? (
          <Typography variant="body1">Scanned bluetooth devices</Typography>
        ) : null}
        {(devices ?? []).map((device, idx) => (
          <Container
            key={`${device.name ?? device.macAddress}-${idx}`}
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: "1rem",
            }}
          >
            <Chip
              variant="outlined"
              color="secondary"
              icon={<BluetoothIcon />}
              label={device.name ?? device.macAddress}
              sx={{ p: "0.5rem" }}
            />
            <Button
              disabled={loading}
              variant="outlined"
              size="small"
              onClick={handleConnectClick(device)}
              endIcon={
                loading && device.macAddress === selectedDevice ? (
                  <CircularProgress color="secondary" size="1rem" />
                ) : null
              }
            >
              Connect
            </Button>
          </Container>
        ))}
      </Container>
      <Snackbar
        message={snackbarSuccessMessage}
        severity="success"
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
      />
      <Snackbar
        message={snackbarErrorMessage}
        severity="error"
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
      />
    </>
  );
};

export default ScannedBluetoothDevices;
