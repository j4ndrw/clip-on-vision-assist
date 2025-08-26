import z from "zod";
import { peripheralApi } from "@/api";

import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import FormField from "@/design-system/form/form-field";
import Button from "@mui/material/Button";
import { useAlertSnackbars } from "@/hooks/use-alert-snackbars";
import { useMicrophoneConfiguration } from "@/hooks/use-microphone-configuration";
import Loading from "@/design-system/loading";
import Form from "@/design-system/form/form";

type MicrophoneConfig = z.infer<
  (typeof peripheralApi)["amendMicrophoneConfiguration"]["requestSchema"]
>["microphoneConfig"];

function MicrophoneConfiguration() {
  const alertSnackbars = useAlertSnackbars();
  const [
    microphoneConfig,
    updateMicrophoneConfig,
    {
      register,
      handleSubmit,
      formState: { errors },
    },
    { handleSaveConfiguration },
  ] = useMicrophoneConfiguration({ alertSnackbars });

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

  if (!microphoneConfig) {
    return <Loading title="Retrieving current microphone config..." />;
  }

  return (
    <Form
      onSubmit={handleSubmit(handleSaveConfiguration)}
      alertSnackbars={alertSnackbars}
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
              errors.microphoneConfig?.audioCaptureConfig?.maxChunks?.message ??
              ""
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
            onChange={handleSilenceDetectionTextFieldChange("minSilenceLenMs")}
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
            value={microphoneConfig.silenceDetectionConfig.silenceThresholdDbfs}
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
    </Form>
  );
}

export default MicrophoneConfiguration;
