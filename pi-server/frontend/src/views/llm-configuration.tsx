import { llmApi } from "@/api";
import FormField from "@/design-system/form/form-field";
import Autocomplete, {
  type AutocompleteChangeReason,
} from "@mui/material/Autocomplete";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Container from "@mui/material/Container";
import InputAdornment from "@mui/material/InputAdornment";
import Typography from "@mui/material/Typography";
import z from "zod";
import { useAlertSnackbars } from "@/hooks/use-alert-snackbars";
import { useLlmConfiguration } from "@/hooks/use-llm-configuration";
import Loading from "@/design-system/loading";
import Form from "@/design-system/form/form";

type LlmConfiguration = z.infer<
  (typeof llmApi)["amendLlmConfiguration"]["requestSchema"]
>;

function LlmConfiguration() {
  const alertSnackbars = useAlertSnackbars();
  const [
    llmConfiguration,
    updateLlmConfiguration,
    {
      register,
      handleSubmit,
      formState: { errors },
    },
    {
      endpointSuggestions,
      availableLlms,
      isEndpointLocal,
      getAvailableLlmsIfEndpointIsValid,
      handleSaveConfiguration,
    },
  ] = useLlmConfiguration({ alertSnackbars });

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

  const handleLlmEndpointChoice = (
    _event: React.SyntheticEvent,
    value: string | null,
    reason: AutocompleteChangeReason,
  ) => {
    switch (reason) {
      case "createOption":
      case "selectOption": {
        if (isEndpointLocal(value ?? ""))
          getAvailableLlmsIfEndpointIsValid(value ?? "");
        return updateLlmConfiguration({ endpoint: value ?? "" });
      }
      case "clear":
      case "removeOption":
        return updateLlmConfiguration({ endpoint: "" });
    }
  };

  if (!llmConfiguration)
    return (
      <>
        <Typography variant="h6">LLM Configuration</Typography>
        <Loading title="Retrieving current LLM config..." />
      </>
    );

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
            (!isEndpointLocal(llmConfiguration.endpoint) &&
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
    </Form>
  );
}

export default LlmConfiguration;
