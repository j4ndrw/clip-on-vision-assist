import { peripheralApi } from "@/api";
import type z from "zod";
import { useReducedState } from "./use-reduced-state";
import { useEffect } from "react";
import { getCurrentMicrophoneConfiguration } from "@/services/peripherals/get-current-microphone-configuration";
import type { useAlertSnackbars } from "./use-alert-snackbars";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { amendMicrophoneConfiguration } from "@/services/peripherals/amend-microphone-configuration";

type MicrophoneConfig = z.infer<
  (typeof peripheralApi)["amendMicrophoneConfiguration"]["requestSchema"]
>["microphoneConfig"];

type Props = {
  alertSnackbars?: ReturnType<typeof useAlertSnackbars>;
};

export const useMicrophoneConfiguration = ({ alertSnackbars }: Props) => {
  const [microphoneConfig, updateMicrophoneConfig] =
    useReducedState<MicrophoneConfig>({
      audioCaptureConfig: { secondsPerChunk: 0, maxChunks: 0 },
      silenceDetectionConfig: { minSilenceLenMs: 0, silenceThresholdDbfs: 0 },
    });
  const form = useForm<
    z.infer<
      (typeof peripheralApi)["amendMicrophoneConfiguration"]["requestSchema"]
    >
  >({
    resolver: zodResolver(
      peripheralApi.amendMicrophoneConfiguration.requestSchema,
    ),
  });

  const handleSaveConfiguration = async (input: {
    microphoneConfig: MicrophoneConfig;
  }) =>
    amendMicrophoneConfiguration({
      input,
      onApiError: (error) => {
        alertSnackbars?.setSnackbarSuccessMessage("");
        alertSnackbars?.setSnackbarErrorMessage(error.message);
      },
      onSuccess: () => {
        alertSnackbars?.setSnackbarSuccessMessage(
          "Saved microphone configuration successfully.",
        );
        alertSnackbars?.setSnackbarErrorMessage("");
      },
    }).promise();

  useEffect(
    () => {
      getCurrentMicrophoneConfiguration({
        onValidationError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
        },
        onApiError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
        },
        onSuccess: (data) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage("");
          updateMicrophoneConfig(data.microphoneConfig);
        },
      }).promise();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  return [
    microphoneConfig,
    updateMicrophoneConfig,
    form,
    { handleSaveConfiguration },
  ] as const;
};
