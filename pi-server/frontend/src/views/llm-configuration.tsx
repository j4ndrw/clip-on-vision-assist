import { llmApi } from "@/api";
import FormField from "@/design-system/form/form-field";
import { amendLlmConfiguration } from "@/services/llm/amend-llm-configuration";
import { getAvailableLlms } from "@/services/llm/get-available-llms";
import { getLlmEndpointSuggestions } from "@/services/llm/get-llm-endpoint-suggestions";
import { debounce } from "@/utils";
import { zodResolver } from "@hookform/resolvers/zod";
import Autocomplete, {
  type AutocompleteChangeReason,
} from "@mui/material/Autocomplete";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Container from "@mui/material/Container";
import InputAdornment from "@mui/material/InputAdornment";
import Snackbar from "@/design-system/snackbar";
import Typography from "@mui/material/Typography";
import { useEffect, useReducer, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { getCurrentLlmConfiguration } from "@/services/llm/get-current-llm-configuration";

type LlmConfiguration = z.infer<
  (typeof llmApi)["amendLlmConfiguration"]["requestSchema"]
>;

const isLocal = (endpoint: string) =>
  endpoint.startsWith("http://localhost") ||
  endpoint.startsWith("https://localhost");

function LlmConfiguration() {
  const [snackbarSuccessMessage, setSnackbarSuccessMessage] = useState("");
  const [snackbarErrorMessage, setSnackbarErrorMessage] = useState("");
  const [llmConfiguration, updateLlmConfiguration] = useReducer<
    LlmConfiguration | null,
    [Partial<LlmConfiguration>]
  >(
    (prev, next) => ({
      ...(prev ?? { model: "", endpoint: "", apiKey: "" }),
      ...next,
    }),
    null,
  );
  const {
    register,
    handleSubmit,
    formState: { errors },
    setError: setFormError,
    clearErrors: clearFormErrors,
  } = useForm<LlmConfiguration>({
    resolver: zodResolver(llmApi.amendLlmConfiguration.requestSchema),
  });

  const [endpointSuggestions, setEndpointSuggestions] = useState<string[]>([]);
  const [availableLlms, setAvailableLlms] = useReducer<
    { value: string[]; pending: boolean },
    [Partial<{ value: string[]; pending: boolean }>]
  >((prev, next) => ({ ...prev, ...next }), {
    value: [],
    pending: false,
  });

  const handleTextFieldChange =
    (
      field: keyof LlmConfiguration,
      options: { sideEffect?: (value: string) => void } = {},
    ): React.ChangeEventHandler<HTMLInputElement | HTMLTextAreaElement> =>
      (e) => {
        updateLlmConfiguration({ [field]: e.target.value });
        options.sideEffect?.(e.target.value);
      };

  const handleLlmChoice = (
    _event: React.SyntheticEvent,
    value: string | null,
    reason: AutocompleteChangeReason,
  ) => {
    if (
      reason === "selectOption" ||
      (reason === "createOption" && availableLlms.value.includes(value ?? ""))
    ) {
      updateLlmConfiguration({ model: value ?? "" });
    }
  };

  const getAvailableLlmsIfEndpointIsValid = debounce(
    async (endpoint: string) => {
      setAvailableLlms({ value: [], pending: true });
      const { error } = z.url().safeParse(endpoint);
      if (!error || isLocal(endpoint) || !!llmConfiguration?.apiKey) {
        return getAvailableLlms({
          onValidationError: (error) => {
            setSnackbarSuccessMessage("");
            setSnackbarErrorMessage(error.message);
          },
          onApiError: (error) => {
            setSnackbarSuccessMessage("");
            setSnackbarErrorMessage(error.message);
          },
          onSuccess: ({ llms }) => {
            setSnackbarErrorMessage("");
            clearFormErrors("endpoint");
            setAvailableLlms({ value: llms, pending: false });
          },
        });
      }

      if (error) {
        setFormError("endpoint", {
          message: error.issues[0]?.message ?? "",
        });
      }
      setAvailableLlms({ value: [], pending: false });
    },
    1000,
  );

  const handleLlmEndpointChoice = (
    _event: React.SyntheticEvent,
    value: string | null,
    reason: AutocompleteChangeReason,
  ) => {
    switch (reason) {
      case "createOption":
      case "selectOption": {
        if (isLocal(value ?? ""))
          getAvailableLlmsIfEndpointIsValid(value ?? "");
        return updateLlmConfiguration({ endpoint: value ?? "" });
      }
      case "clear":
      case "removeOption":
        return updateLlmConfiguration({ endpoint: "" });
    }
  };

  const handleSaveConfiguration = async (llmConfiguration: LlmConfiguration) =>
    amendLlmConfiguration({
      input: llmConfiguration,
      onApiError: (error) => {
        setSnackbarSuccessMessage("");
        setSnackbarErrorMessage(error.message);
      },
      onSuccess: () => {
        setSnackbarErrorMessage("");
        setSnackbarSuccessMessage("Saved LLM configuration successfully.");
      },
    });

  const handleSuccessSnackbarClose = () => {
    setSnackbarSuccessMessage("");
  };

  const handleErrorSnackbarClose = () => {
    setSnackbarErrorMessage("");
  };

  useEffect(() => {
    getLlmEndpointSuggestions({
      onValidationError: (error) => {
        setSnackbarSuccessMessage("");
        setSnackbarErrorMessage(error.message);
      },
      onApiError: (error) => {
        setSnackbarSuccessMessage("");
        setSnackbarErrorMessage(error.message);
      },
      onSuccess: ({ endpointSuggestions }) => {
        setSnackbarErrorMessage("");
        setEndpointSuggestions(endpointSuggestions);
      },
    });
  }, []);

  useEffect(
    () => {
      getCurrentLlmConfiguration({
        onValidationError: (error) => {
          setSnackbarSuccessMessage("");
          setSnackbarErrorMessage(error.message);
        },
        onApiError: (error) => {
          setSnackbarSuccessMessage("");
          setSnackbarErrorMessage(error.message);
        },
        onSuccess: ({ llmConfig }) => {
          setSnackbarErrorMessage("");
          updateLlmConfiguration({
            ...llmConfig,
            endpoint: llmConfig.endpoint ?? "",
          });
          if (llmConfig.endpoint) {
            getAvailableLlmsIfEndpointIsValid(llmConfig.endpoint);
          }
        },
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  if (!llmConfiguration)
    return (
      <>
        <Typography variant="h6">LLM Configuration</Typography>
        <Container
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: "1rem",
          }}
        >
          <Typography variant="overline">
            Retrieving current LLM config...
          </Typography>
          <CircularProgress color="info" size="1rem" />
        </Container>
      </>
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
            gap: "1rem",
            width: "100%",
          }}
        >
          <Typography variant="h6">LLM Configuration</Typography>
          <Autocomplete
            freeSolo
            fullWidth
            options={endpointSuggestions}
            {...register("endpoint")}
            onChange={handleLlmEndpointChoice}
            value={llmConfiguration?.endpoint ?? ""}
            renderInput={(params) => (
              <FormField
                {...params}
                onChange={handleTextFieldChange("endpoint", {
                  sideEffect: (value) => {
                    updateLlmConfiguration({ model: "" });
                    getAvailableLlmsIfEndpointIsValid(value ?? "");
                  },
                })}
                label="LLM Backend to use"
                errorMessage={errors.endpoint?.message ?? ""}
              />
            )}
          />
          <FormField
            fullWidth
            disabled={!llmConfiguration?.endpoint}
            {...register("apiKey")}
            onChange={handleTextFieldChange("apiKey", {
              sideEffect: () =>
                getAvailableLlmsIfEndpointIsValid(llmConfiguration.endpoint),
            })}
            label={
              llmConfiguration?.endpoint
                ? `API Key for ${llmConfiguration?.endpoint}`
                : "API Key"
            }
            value={llmConfiguration.apiKey ?? ""}
            errorMessage={errors.apiKey?.message ?? ""}
          />
          <Autocomplete
            options={availableLlms.value}
            fullWidth
            disabled={
              !llmConfiguration.endpoint ||
              (!isLocal(llmConfiguration.endpoint) &&
                !llmConfiguration.apiKey) ||
              (availableLlms.value.length === 0 && !llmConfiguration.model) ||
              availableLlms.pending
            }
            {...register("model")}
            onChange={handleLlmChoice}
            value={llmConfiguration?.model ?? ""}
            renderInput={(params) => (
              <FormField
                {...params}
                label={
                  <span style={{ display: "flex" }}>
                    LLM to use
                    {availableLlms.pending ? (
                      <InputAdornment position="end">
                        <CircularProgress size="1rem" />
                      </InputAdornment>
                    ) : null}
                  </span>
                }
                errorMessage={errors.model?.message ?? ""}
              />
            )}
          />
          <Button
            type="submit"
            disabled={
              !llmConfiguration.model ||
              !llmConfiguration.endpoint ||
              availableLlms.pending
            }
          >
            Save Configuration
          </Button>
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

export default LlmConfiguration;
