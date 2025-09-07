import { computeServerApi } from "@/api";
import z from "zod";
import { useReducedState } from "./use-reduced-state";
import type { useAlertSnackbars } from "./use-alert-snackbars";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { amendComputeServerConfiguration } from "@/services/compute-server/amend-compute-server-configuration";
import { useEffect } from "react";
import { getCurrentComputeServerConfiguration } from "@/services/compute-server/get-current-compute-server-configuration";

type ComputeServerConfiguration = z.infer<
  (typeof computeServerApi)["amendComputeServerConfiguration"]["requestSchema"]
>;

type Props = {
  alertSnackbars?: ReturnType<typeof useAlertSnackbars>;
};

export const useComputeServerConfiguration = ({ alertSnackbars }: Props) => {
  const [computeServerConfiguration, updateComputeServerConfiguration] =
    useReducedState<ComputeServerConfiguration>({ endpoint: "", apiKey: "" });

  const form = useForm<ComputeServerConfiguration>({
    resolver: zodResolver(
      computeServerApi.amendComputeServerConfiguration.requestSchema,
    ),
  });

  const handleSaveConfiguration = async (input: ComputeServerConfiguration) =>
    amendComputeServerConfiguration({
      input: computeServerConfiguration ?? input,
      onApiError: (error) => {
        alertSnackbars?.setSnackbarSuccessMessage("");
        alertSnackbars?.setSnackbarErrorMessage(error.message);
      },
      onSuccess: () => {
        alertSnackbars?.setSnackbarErrorMessage("");
        alertSnackbars?.setSnackbarSuccessMessage(
          "Saved Compute Server configuration successfully.",
        );
      },
    }).promise();

  useEffect(
    () => {
      getCurrentComputeServerConfiguration({
        onValidationError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
        },
        onApiError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
        },
        onSuccess: ({ computeServerConfig }) => {
          alertSnackbars?.setSnackbarErrorMessage("");
          updateComputeServerConfiguration({
            ...computeServerConfig,
            endpoint: computeServerConfig.endpoint ?? "",
          });
        },
      }).promise();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  return [
    computeServerConfiguration,
    updateComputeServerConfiguration,
    form,
    {
      handleSaveConfiguration,
    },
  ] as const;
};
