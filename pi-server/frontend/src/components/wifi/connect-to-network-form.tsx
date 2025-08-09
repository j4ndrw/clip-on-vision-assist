import { connectToNetwork } from "@/services/wifi/connect-to-network";
import { useWifiStore } from "@/store";
import {
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
} from "@mui/icons-material";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import Box from "@mui/material/Box";
import { useEffect, useRef, useState } from "react";
import FilledInput from "@mui/material/FilledInput";
import Container from "@mui/material/Container";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Modal from "@mui/material/Modal";
import { sleep } from "@/utils";
import { healthCheck } from "@/services/health-check";
import Typography from "@mui/material/Typography";
import Snackbar from "@/design-system/snackbar";

function ConnectToNetwork() {
  const healthCheckInterval = useRef<NodeJS.Timeout | null>(null);
  const { selectedNetwork, selectNetwork } = useWifiStore();

  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarErrorMessage, setSnackbarErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleClickShowPassword = () => {
    setShowPassword((prev) => !prev);
  };

  const handleCancel = () => {
    setSnackbarMessage("");
    setSnackbarErrorMessage("");
    selectNetwork(null);
    setLoading(false);
    setPassword("");
  };

  const handlePasswordInputOnChange: React.ChangeEventHandler<
    HTMLTextAreaElement | HTMLInputElement
  > = (e) => {
    setPassword(e.target.value);
  };

  const handleConnectClick = async () => {
    if (!selectedNetwork) return;

    setLoading(true);

    // FIXME: We really need an abort controller here. Otherwise, when the network comes back, we'll
    // spam the server with no-ops unnecessarily...
    // Something should be implemented in the `@/api` abstraction layer so that abort controllers
    // are supported in a declarative way.
    await Promise.all([
      connectToNetwork({
        input: { ssid: selectedNetwork.ssid, password },
        onApiError: (error) => {
          setLoading(false);
          setSnackbarErrorMessage(error.message);
          setSnackbarMessage("");
          selectNetwork(null);
          setPassword("");
          if (healthCheckInterval.current) clearInterval(healthCheckInterval.current)
          return;
        },
        onSuccess: () => { }, // Can be ignored - we'll use the healthcheck to gauge if this worked. Hacky, but works for the moment.
      }),
      (async () => {
        await sleep(6000);
        healthCheckInterval.current = setInterval(
          async () =>
            await healthCheck({
              onApiError: () => { }, // Can be skipped since we're just listening to see when the connection comes back
              onSuccess: () => {
                if (healthCheckInterval.current) {
                  clearInterval(healthCheckInterval.current);
                  healthCheckInterval.current = null;
                }
                setLoading(false);
                setSnackbarErrorMessage("");
                setSnackbarMessage(
                  `Successfully connected to Wi-Fi \`${selectedNetwork.ssid}\``,
                );
                selectNetwork(null);
                setPassword("");
              },
            }),
          1000,
        );
      })(),
    ]);
  };

  useEffect(() => {
    return () => {
      if (healthCheckInterval.current)
        clearInterval(healthCheckInterval.current);
    };
  }, []);

  return (
    <>
      <Modal open={!!selectedNetwork}>
        {selectedNetwork ? (
          <Box
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              width: 400,
              bgcolor: "background.paper",
              border: "2px solid #FFF",
              boxShadow: 24,
              pt: 2,
              px: 4,
              pb: 3,
            }}
          >
            <Container
              sx={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
                gap: "1rem",
              }}
            >
              <Chip
                variant="outlined"
                color="secondary"
                label={selectedNetwork!.ssid}
              />
              <FilledInput
                sx={{ display: "flex", justifyContent: "center" }}
                type={showPassword ? "text" : "password"}
                onChange={handlePasswordInputOnChange}
                endAdornment={
                  <InputAdornment position="end">
                    <IconButton
                      size="small"
                      onClick={handleClickShowPassword}
                      edge="end"
                    >
                      {showPassword ? (
                        <VisibilityOffIcon />
                      ) : (
                        <VisibilityIcon />
                      )}
                    </IconButton>
                  </InputAdornment>
                }
                placeholder="Enter Wi-Fi Password..."
              />
              <Container
                sx={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  gap: "2rem",
                }}
              >
                <Button
                  variant="outlined"
                  color="secondary"
                  size="small"
                  disabled={loading}
                  onClick={handleCancel}
                >
                  Cancel
                </Button>
                <Button
                  variant="outlined"
                  color="primary"
                  disabled={!password || loading}
                  size="small"
                  onClick={handleConnectClick}
                  endIcon={
                    loading ? (
                      <CircularProgress color="secondary" size="1rem" />
                    ) : null
                  }
                >
                  Connect
                </Button>
              </Container>
              {loading ? (
                <Typography variant="body1">
                  NOTE: You may have to manually reconnect to the hotspot if
                  disconnected.
                </Typography>
              ) : null}
            </Container>
          </Box>
        ) : (
          <></>
        )}
      </Modal>
      <Snackbar
        message={snackbarMessage}
        severity="success"
        autoHideDuration={6000}
        onClose={handleCancel}
      />
      <Snackbar
        message={snackbarErrorMessage}
        severity="error"
        autoHideDuration={6000}
        onClose={handleCancel}
      />
    </>
  );
}

export default ConnectToNetwork;
