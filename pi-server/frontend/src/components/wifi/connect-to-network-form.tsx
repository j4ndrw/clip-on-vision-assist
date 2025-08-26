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
import { useState } from "react";
import FilledInput from "@mui/material/FilledInput";
import Container from "@mui/material/Container";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Modal from "@mui/material/Modal";
import Typography from "@mui/material/Typography";
import Snackbar from "@/design-system/snackbar";
import { useAlertSnackbars } from "@/hooks/use-alert-snackbars";
import { useListenUntilBackOnline } from "@/hooks/use-listen-until-back-online";

function ConnectToNetwork() {
  const {
    snackbarSuccessMessage,
    snackbarErrorMessage,
    setSnackbarSuccessMessage,
    setSnackbarErrorMessage,
  } = useAlertSnackbars();
  const { selectedNetwork, selectNetwork } = useWifiStore();

  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const { listenUntilBackOnline, healthCheckInterval } =
    useListenUntilBackOnline({
      preflightDelayMs: 6000,
      checkEveryMs: 1000,
      onSuccess: () => {
        setLoading(false);
        setSnackbarErrorMessage("");
        if (selectedNetwork) {
          setSnackbarSuccessMessage(
            `Successfully connected to Wi-Fi \`${selectedNetwork.ssid}\``,
          );
        }
        selectNetwork(null);
        setPassword("");
      },
    });

  const handleClickShowPassword = () => {
    setShowPassword((prev) => !prev);
  };

  const handleCancel = () => {
    setSnackbarSuccessMessage("");
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

    await Promise.all([
      connectToNetwork({
        input: { ssid: selectedNetwork.ssid, password },
        onApiError: (error) => {
          setLoading(false);
          setSnackbarErrorMessage(error.message);
          setSnackbarSuccessMessage("");
          selectNetwork(null);
          setPassword("");
          if (healthCheckInterval.current)
            clearInterval(healthCheckInterval.current);
        },
        onSuccess: () => { }, // Can be ignored - we'll use the healthcheck to gauge if this worked. Hacky, but works for the moment.
      }).promise(),
      listenUntilBackOnline(),
    ]);
  };

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
        message={snackbarSuccessMessage}
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
