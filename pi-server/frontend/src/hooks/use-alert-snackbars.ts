import { useState } from "react";

export const useAlertSnackbars = () => {
  const [snackbarSuccessMessage, setSnackbarSuccessMessage] = useState("");
  const [snackbarErrorMessage, setSnackbarErrorMessage] = useState("");

  const handleSuccessSnackbarClose = () => {
    setSnackbarSuccessMessage("");
  };

  const handleErrorSnackbarClose = () => {
    setSnackbarErrorMessage("");
  };

  return {
    snackbarSuccessMessage,
    snackbarErrorMessage,
    setSnackbarSuccessMessage,
    setSnackbarErrorMessage,
    handleSuccessSnackbarClose,
    handleErrorSnackbarClose,
  };
};
