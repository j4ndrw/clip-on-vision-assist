import { useEffect, useReducer, useState } from "react";
import z from "zod";
import { peripheralApi } from "@/api";
import { amendMicrophoneConfiguration } from "@/services/peripherals/amend-microphone-configuration";

import Container from "@mui/material/Container";
import { getCurrentMicrophoneConfiguration } from "@/services/peripherals/get-current-microphone-configuration";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import FormField from "@/design-system/form/form-field";
import Button from "@mui/material/Button";
import Snackbar from "@/design-system/snackbar";

type MicrophoneConfig = z.infer<
  (typeof peripheralApi)["amendMicrophoneConfiguration"]["requestSchema"]
>["microphoneConfig"];

function MicrophoneConfiguration() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<
    z.infer<
      (typeof peripheralApi)["amendMicrophoneConfiguration"]["requestSchema"]
    >
  >({
    resolver: zodResolver(
      peripheralApi.amendMicrophoneConfiguration.requestSchema,
    ),
  });
  const [microphoneConfig, updateMicrophoneConfig] = useReducer<
    MicrophoneConfig | null,
    [Partial<MicrophoneConfig>]
  >(
    (prev, next) => ({
      ...(prev ?? {
        audioCaptureConfig: { secondsPerChunk: 0, maxChunks: 0 },
        silenceDetectionConfig: { minSilenceLenMs: 0, silenceThresholdDbfs: 0 },
      }),
      ...next,
    }),
    null,
  );
  const [snackbarSuccessMessage, setSnackbarSuccessMessage] = useState("");
  const [snackbarErrorMessage, setSnackbarErrorMessage] = useState("");

  const handleSaveConfiguration = async ({
    microphoneConfig,
  }: {
    microphoneConfig: MicrophoneConfig;
  }) => {
    await amendMicrophoneConfiguration({
      input: { microphoneConfig },
      onApiError: (error) => {
        setSnackbarSuccessMessage("");
        setSnackbarErrorMessage(error.message);
      },
      onSuccess: () => {
        setSnackbarSuccessMessage(
          "Saved microphone configuration successfully.",
        );
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

  const handleAudioCaptureTextFieldChange =
    (
      field: keyof MicrophoneConfig["audioCaptureConfig"],
    ): React.ChangeEventHandler<HTMLInputElement | HTMLTextAreaElement> =>
      (e) => {
        updateMicrophoneConfig({
          audioCaptureConfig: {
            secondsPerChunk:
              microphoneConfig?.audioCaptureConfig.secondsPerChunk ?? 0,
            maxChunks: microphoneConfig?.audioCaptureConfig.maxChunks ?? 0,
            [field]: e.target.value,
          },
        });
      };

  const handleSilenceDetectionTextFieldChange =
    (
      field: keyof MicrophoneConfig["silenceDetectionConfig"],
    ): React.ChangeEventHandler<HTMLInputElement | HTMLTextAreaElement> =>
      (e) => {
        updateMicrophoneConfig({
          silenceDetectionConfig: {
            minSilenceLenMs:
              microphoneConfig?.silenceDetectionConfig?.minSilenceLenMs ?? 0,
            silenceThresholdDbfs:
              microphoneConfig?.silenceDetectionConfig.silenceThresholdDbfs ?? 0,
            [field]: e.target.value,
          },
        });
      };

  useEffect(() => {
    (async () => {
      await getCurrentMicrophoneConfiguration({
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
          updateMicrophoneConfig(data.microphoneConfig);
        },
      });
    })();
  }, []);

  if (!microphoneConfig)
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
          Retrieving current microphone config...
        </Typography>
        <CircularProgress color="info" size="1rem" />
      </Container>
    );

  return (
    <>
      <form
        style={{ width: "100%" }}
        onSubmit={handleSubmit(handleSaveConfiguration)}
      >
        <Container
          sx={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            gap: "2rem",
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
            <Typography variant="subtitle1">
              Audio Capture Configuration
            </Typography>
            <FormField
              fullWidth
              type="number"
              label="Seconds per chunk"
              {...register(
                "microphoneConfig.audioCaptureConfig.secondsPerChunk",
                {
                  valueAsNumber: true,
                },
              )}
              value={microphoneConfig.audioCaptureConfig.secondsPerChunk}
              onChange={handleAudioCaptureTextFieldChange("secondsPerChunk")}
              errorMessage={
                errors.microphoneConfig?.audioCaptureConfig?.secondsPerChunk
                  ?.message ?? ""
              }
            />
            <FormField
              fullWidth
              type="number"
              label="Max chunks to record at a time"
              {...register("microphoneConfig.audioCaptureConfig.maxChunks", {
                valueAsNumber: true,
              })}
              value={microphoneConfig.audioCaptureConfig.maxChunks}
              onChange={handleAudioCaptureTextFieldChange("maxChunks")}
              errorMessage={
                errors.microphoneConfig?.audioCaptureConfig?.maxChunks
                  ?.message ?? ""
              }
            />
          </Container>
          <Container
            sx={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
              gap: "1rem",
            }}
          >
            <Typography variant="subtitle1">
              Silence Detection Configuration
            </Typography>
            <FormField
              fullWidth
              type="number"
              label="Minimum silence length (ms)"
              {...register(
                "microphoneConfig.silenceDetectionConfig.minSilenceLenMs",
                {
                  valueAsNumber: true,
                },
              )}
              value={microphoneConfig.silenceDetectionConfig.minSilenceLenMs}
              onChange={handleSilenceDetectionTextFieldChange(
                "minSilenceLenMs",
              )}
              errorMessage={
                errors.microphoneConfig?.silenceDetectionConfig?.minSilenceLenMs
                  ?.message ?? ""
              }
            />
            <FormField
              fullWidth
              type="number"
              label="Silence threshold (decibels)"
              {...register(
                "microphoneConfig.silenceDetectionConfig.silenceThresholdDbfs",
                {
                  valueAsNumber: true,
                },
              )}
              value={
                microphoneConfig.silenceDetectionConfig.silenceThresholdDbfs
              }
              onChange={handleSilenceDetectionTextFieldChange(
                "silenceThresholdDbfs",
              )}
              errorMessage={
                errors.microphoneConfig?.silenceDetectionConfig
                  ?.silenceThresholdDbfs?.message ?? ""
              }
            />
          </Container>
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

export default MicrophoneConfiguration;
