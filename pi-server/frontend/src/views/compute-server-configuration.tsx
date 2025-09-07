import { computeServerApi } from "@/api";
import FormField from "@/design-system/form/form-field";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import z from "zod";
import { useAlertSnackbars } from "@/hooks/use-alert-snackbars";
import { useComputeServerConfiguration } from "@/hooks/use-compute-server-configuration";
import Form from "@/design-system/form/form";
import Loading from "@/design-system/loading";

type ComputeServerConfiguration = z.infer<
  (typeof computeServerApi)["amendComputeServerConfiguration"]["requestSchema"]
>;

function ComputeServerConfiguration() {
  const alertSnackbars = useAlertSnackbars();
  const [
    computeServerConfiguration,
    updateComputeServerConfiguration,
    {
      register,
      handleSubmit,
      formState: { errors },
    },
    { handleSaveConfiguration },
  ] = useComputeServerConfiguration({ alertSnackbars });

  const handleTextFieldChange =
    (
      field: keyof ComputeServerConfiguration,
    ): React.ChangeEventHandler<HTMLInputElement | HTMLTextAreaElement> =>
      (e) => {
        updateComputeServerConfiguration({ [field]: e.target.value });
      };

  if (!computeServerConfiguration)
    return (
      <>
        <Typography variant="h6">Compute Server Configuration</Typography>
        <Loading title="Retrieving current compute server config..." />
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
        <Typography variant="h6">Compute Server Configuration</Typography>
        <FormField
          fullWidth
          {...register("endpoint")}
          onChange={handleTextFieldChange("endpoint")}
          label="Endpoint"
          value={computeServerConfiguration.endpoint ?? ""}
          errorMessage={errors.endpoint?.message ?? ""}
        />
        <FormField
          fullWidth
          {...register("apiKey")}
          onChange={handleTextFieldChange("apiKey")}
          label="API Key"
          value={computeServerConfiguration?.apiKey ?? ""}
          errorMessage={errors.apiKey?.message ?? ""}
        />
        <Button
          type="submit"
          disabled={
            !computeServerConfiguration?.apiKey ||
            !computeServerConfiguration.endpoint
          }
        >
          Save Configuration
        </Button>
      </Container>
    </Form>
  );
}

export default ComputeServerConfiguration;
