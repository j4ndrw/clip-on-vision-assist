import { useEffect, useReducer, useState } from "react";
import z from "zod";
import { peripheralApi } from "@/api";
import { amendCameraConfiguration } from "@/services/peripherals/amend-camera-configuration";

import Container from "@mui/material/Container";
import { getCurrentCameraConfiguration } from "@/services/peripherals/get-current-camera-configuration";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import FormField from "@/design-system/form/form-field";
import Button from "@mui/material/Button";
import Snackbar from "@/design-system/snackbar";

type CameraConfig = z.infer<
  (typeof peripheralApi)["amendCameraConfiguration"]["requestSchema"]
>["cameraConfig"];

function CameraConfiguration() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<
    z.infer<(typeof peripheralApi)["amendCameraConfiguration"]["requestSchema"]>
  >({
    resolver: zodResolver(peripheralApi.amendCameraConfiguration.requestSchema),
  });
  const [cameraConfig, updateCameraConfig] = useReducer<
    CameraConfig | null,
    [Partial<CameraConfig>]
  >(
    (prev, next) => ({
      ...(prev ?? { numFramesToCapture: 0, fps: 0, waitForNextBatchFactor: 0 }),
      ...next,
    }),
    null,
  );
  const [snackbarSuccessMessage, setSnackbarSuccessMessage] = useState("");
  const [snackbarErrorMessage, setSnackbarErrorMessage] = useState("");

  const handleSaveConfiguration = async ({
    cameraConfig,
  }: {
    cameraConfig: CameraConfig;
  }) => {
    await amendCameraConfiguration({
      input: { cameraConfig },
      onApiError: (error) => {
        setSnackbarSuccessMessage("");
        setSnackbarErrorMessage(error.message);
      },
      onSuccess: () => {
        setSnackbarSuccessMessage("Saved camera configuration successfully.");
        setSnackbarErrorMessage("");
      },
    });
  };

  const handleSuccessSnackbarClose = () => {
    setSnackbarSuccessMessage("");
  };

  const handleErrorSnackbarClose = () => {
    setSnackbarErrorMessage("");
  };

  const handleTextFieldChange =
    (
      field: keyof CameraConfig,
    ): React.ChangeEventHandler<HTMLInputElement | HTMLTextAreaElement> =>
      (e) => {
        updateCameraConfig({ [field]: e.target.value });
      };

  useEffect(() => {
    (async () => {
      await getCurrentCameraConfiguration({
        onValidationError: (error) => {
          setSnackbarSuccessMessage("");
          setSnackbarErrorMessage(error.message);
        },
        onApiError: (error) => {
          setSnackbarSuccessMessage("");
          setSnackbarErrorMessage(error.message);
        },
        onSuccess: (data) => {
          setSnackbarSuccessMessage("");
          setSnackbarErrorMessage("");
          updateCameraConfig(data.cameraConfig);
        },
      });
    })();
  }, []);

  if (!cameraConfig)
    return (
      <Container
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: "1rem",
        }}
      >
        <Typography variant="overline">
          Retrieving current camera config...
        </Typography>
        <CircularProgress color="info" size="1rem" />
      </Container>
    );

  return (
    <>
      <form onSubmit={handleSubmit(handleSaveConfiguration)}>
        <Container
          sx={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            gap: "1rem",
          }}
        >
          <FormField
            fullWidth
            type="number"
            label="Number of frames to capture"
            {...register("cameraConfig.numFramesToCapture", {
              valueAsNumber: true,
            })}
            value={cameraConfig.numFramesToCapture}
            onChange={handleTextFieldChange("numFramesToCapture")}
            errorMessage={
              errors.cameraConfig?.numFramesToCapture?.message ?? ""
            }
          />
          <FormField
            fullWidth
            type="number"
            label="Frames per second"
            {...register("cameraConfig.fps", { valueAsNumber: true })}
            value={cameraConfig.fps}
            onChange={handleTextFieldChange("fps")}
            errorMessage={errors.cameraConfig?.fps?.message ?? ""}
          />
          <FormField
            fullWidth
            type="number"
            label="Wait for next frame batch factor"
            {...register("cameraConfig.waitForNextBatchFactor", {
              valueAsNumber: true,
            })}
            value={cameraConfig.waitForNextBatchFactor}
            onChange={handleTextFieldChange("waitForNextBatchFactor")}
            errorMessage={
              errors.cameraConfig?.waitForNextBatchFactor?.message ?? ""
            }
          />
          <Button type="submit">Save Configuration</Button>
        </Container>
      </form>
      <Snackbar
        message={snackbarSuccessMessage}
        severity="success"
        onClose={handleSuccessSnackbarClose}
      />
      <Snackbar
        message={snackbarErrorMessage}
        severity="error"
        onClose={handleErrorSnackbarClose}
      />
    </>
  );
}

export default CameraConfiguration;
