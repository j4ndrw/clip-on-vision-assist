import Alert from "@mui/material/Alert";
import MUISnackbar, { type SnackbarCloseReason } from "@mui/material/Snackbar";

const DEFAULT_AUTO_HIDE_DURATION_MS = 6000;

type Props = {
  message: string;
  severity: NonNullable<React.ComponentProps<typeof Alert>["severity"]>;
  autoHideDuration?: number | 'none';
  onClose?: () => void;
};

function Snackbar({ message, severity, autoHideDuration = DEFAULT_AUTO_HIDE_DURATION_MS, onClose }: Props) {
  const handleClose = (
    _event: React.SyntheticEvent | Event,
    reason?: SnackbarCloseReason,
  ) => {
    if (reason === "clickaway") {
      return;
    }

    onClose?.();
  };

  return (
      <MUISnackbar
        open={!!message}
        autoHideDuration={autoHideDuration === 'none' ? undefined : autoHideDuration}
        onClose={onClose ? handleClose : undefined}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={onClose ? handleClose : undefined}
          severity={severity}
          variant="outlined"
          sx={{ width: "100%" }}
        >
          {message}
        </Alert>
      </MUISnackbar>
  );
}

export default Snackbar;
