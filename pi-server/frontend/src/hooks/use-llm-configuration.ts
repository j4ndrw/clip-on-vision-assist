import { llmApi } from "@/api";
import z from "zod";
import { useNonNullReducedState, useReducedState } from "./use-reduced-state";
import { useEffect, useState } from "react";
import type { useAlertSnackbars } from "./use-alert-snackbars";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { getCurrentLlmConfiguration } from "@/services/llm/get-current-llm-configuration";
import { amendLlmConfiguration } from "@/services/llm/amend-llm-configuration";
import { debounce, withRetry } from "@/utils";
import { getAvailableLlms } from "@/services/llm/get-available-llms";
import { getLlmEndpointSuggestions } from "@/services/llm/get-llm-endpoint-suggestions";

type LlmConfiguration = z.infer<
  (typeof llmApi)["amendLlmConfiguration"]["requestSchema"]
>;

const isEndpointLocal = (endpoint: string) =>
  endpoint.startsWith("http://localhost") ||
  endpoint.startsWith("https://localhost");

type Props = {
  alertSnackbars?: ReturnType<typeof useAlertSnackbars>;
};

export const useLlmConfiguration = ({ alertSnackbars }: Props) => {
  const [llmConfiguration, updateLlmConfiguration] =
    useReducedState<LlmConfiguration>({ model: "", endpoint: "", apiKey: "" });

  const [availableLlms, setAvailableLlms] = useNonNullReducedState<{
    value: string[];
    pending: boolean;
  }>({
    value: [],
    pending: false,
  });

  const [endpointSuggestions, setEndpointSuggestions] = useState<string[]>([]);

  const form = useForm<LlmConfiguration>({
    resolver: zodResolver(llmApi.amendLlmConfiguration.requestSchema),
  });

  const handleSaveConfiguration = async (input: LlmConfiguration) =>
    amendLlmConfiguration({
      input: llmConfiguration ?? input,
      onApiError: (error) => {
        alertSnackbars?.setSnackbarSuccessMessage("");
        alertSnackbars?.setSnackbarErrorMessage(error.message);
      },
      onSuccess: () => {
        alertSnackbars?.setSnackbarErrorMessage("");
        alertSnackbars?.setSnackbarSuccessMessage(
          "Saved LLM configuration successfully.",
        );
      },
    }).promise();

  // FIXME: Invalidate previous request or increase debounce time.
  // Otherwise, race conditions can happen.
  const getAvailableLlmsIfEndpointIsValid = debounce(
    // FIXME: For some reason, if the compute server is on, but the LLM backend isn't
    // this doesn't trigger a retry :(
    withRetry(
      (triggerRetry) => async (endpoint: string) => {
        setAvailableLlms({ value: [], pending: true });
        const { error } = z.url().safeParse(endpoint);
        if (!error || isEndpointLocal(endpoint) || !!llmConfiguration?.apiKey) {
          return getAvailableLlms({
            queryParams: { endpoint },
            onValidationError: (error) => {
              alertSnackbars?.setSnackbarSuccessMessage("");
              alertSnackbars?.setSnackbarErrorMessage(error.message);
            },
            onApiError: (error) => {
              alertSnackbars?.setSnackbarSuccessMessage("");
              alertSnackbars?.setSnackbarErrorMessage(error.message);
              triggerRetry();
            },
            onSuccess: ({ llms }) => {
              alertSnackbars?.setSnackbarErrorMessage("");
              form.clearErrors("endpoint");
              setAvailableLlms({ value: llms, pending: false });
            },
          }).promise();
        }

        if (error) {
          form.setError("endpoint", {
            message: error.issues[0]?.message ?? "",
          });
        }
        setAvailableLlms({ value: [], pending: false });
      },
      5000,
    ),
    1000,
  );

  useEffect(
    () => {
      getCurrentLlmConfiguration({
        onValidationError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
        },
        onApiError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
        },
        onSuccess: ({ llmConfig }) => {
          alertSnackbars?.setSnackbarErrorMessage("");
          updateLlmConfiguration({
            ...llmConfig,
            endpoint: llmConfig.endpoint ?? "",
          });
          if (llmConfig.endpoint) {
            getAvailableLlmsIfEndpointIsValid(llmConfig.endpoint);
          }
        },
      }).promise();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useEffect(
    () => {
      getLlmEndpointSuggestions({
        onValidationError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
        },
        onApiError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
        },
        onSuccess: ({ endpointSuggestions }) => {
          alertSnackbars?.setSnackbarErrorMessage("");
          setEndpointSuggestions(endpointSuggestions);
        },
      }).promise();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  return [
    llmConfiguration,
    updateLlmConfiguration,
    form,
    {
      endpointSuggestions,
      availableLlms,
      getAvailableLlmsIfEndpointIsValid,
      isEndpointLocal,
      handleSaveConfiguration,
    },
  ] as const;
};
