import { hotspotApi } from "@/api";
import type z from "zod";
import { useReducedState } from "./use-reduced-state";
import { useEffect } from "react";
import { getCurrentHotspotConfiguration } from "@/services/hotspot/get-current-hotspot-configuration";
import type { useAlertSnackbars } from "./use-alert-snackbars";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { amendHotspotConfiguration } from "@/services/hotspot/amend-hotspot-configuration";

type HotspotConfig = z.infer<
  (typeof hotspotApi)["amendHotspotConfiguration"]["requestSchema"]
>["hotspotConfig"];

type Props = {
  alertSnackbars?: ReturnType<typeof useAlertSnackbars>;
};

export const useHotspotConfiguration = ({ alertSnackbars }: Props) => {
  const [hotspotConfig, updateHotspotConfig] = useReducedState<HotspotConfig>({
    ssid: "",
    password: "",
  });
  const form = useForm<
    z.infer<(typeof hotspotApi)["amendHotspotConfiguration"]["requestSchema"]>
  >({
    resolver: zodResolver(hotspotApi.amendHotspotConfiguration.requestSchema),
  });

  const handleSaveConfiguration = async (input: {
    hotspotConfig: HotspotConfig;
  }) =>
    amendHotspotConfiguration({
      input,
      onApiError: (error) => {
        alertSnackbars?.setSnackbarSuccessMessage("");
        alertSnackbars?.setSnackbarErrorMessage(error.message);
      },
      onSuccess: () => {
        alertSnackbars?.setSnackbarSuccessMessage(
          "Saved hotspot configuration successfully.",
        );
        alertSnackbars?.setSnackbarErrorMessage("");
      },
    }).promise();

  useEffect(
    () => {
      getCurrentHotspotConfiguration({
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
          updateHotspotConfig(data.hotspotConfig);
        },
      }).promise();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  return [
    hotspotConfig,
    updateHotspotConfig,
    form,
    { handleSaveConfiguration },
  ] as const;
};
