import { peripheralApi } from "@/api";
import type z from "zod";
import { useReducedState } from "./use-reduced-state";
import { useEffect } from "react";
import type { useAlertSnackbars } from "./use-alert-snackbars";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { amendCameraConfiguration } from "@/services/peripherals/amend-camera-configuration";
import { getCurrentCameraConfiguration } from "@/services/peripherals/get-current-camera-configuration";

type CameraConfig = z.infer<
  (typeof peripheralApi)["amendCameraConfiguration"]["requestSchema"]
>["cameraConfig"];

type Props = {
  alertSnackbars?: ReturnType<typeof useAlertSnackbars>;
};

export const useCameraConfiguration = ({ alertSnackbars }: Props) => {
  const [cameraConfig, updateCameraConfig] = useReducedState<CameraConfig>({
    numFramesToCapture: 0,
    fps: 0,
    waitForNextBatchFactor: 0,
  });

  const form = useForm<
    z.infer<(typeof peripheralApi)["amendCameraConfiguration"]["requestSchema"]>
  >({
    resolver: zodResolver(peripheralApi.amendCameraConfiguration.requestSchema),
  });

  const handleSaveConfiguration = async (input: {
    cameraConfig: CameraConfig;
  }) =>
    amendCameraConfiguration({
      input,
      onApiError: (error) => {
        alertSnackbars?.setSnackbarSuccessMessage("");
        alertSnackbars?.setSnackbarErrorMessage(error.message);
      },
      onSuccess: () => {
        alertSnackbars?.setSnackbarSuccessMessage(
          "Saved camera configuration successfully.",
        );
        alertSnackbars?.setSnackbarErrorMessage("");
      },
    }).promise();

  useEffect(
    () => {
      getCurrentCameraConfiguration({
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
          updateCameraConfig(data.cameraConfig);
        },
      }).promise();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  return [
    cameraConfig,
    updateCameraConfig,
    form,
    { handleSaveConfiguration },
  ] as const;
};
