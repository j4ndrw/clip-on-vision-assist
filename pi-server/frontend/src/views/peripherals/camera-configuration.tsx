import z from "zod";
import { peripheralApi } from "@/api";

import Container from "@mui/material/Container";
import FormField from "@/design-system/form/form-field";
import Button from "@mui/material/Button";
import { useAlertSnackbars } from "@/hooks/use-alert-snackbars";
import { useCameraConfiguration } from "@/hooks/use-camera-configuration";
import Form from "@/design-system/form/form";
import Loading from "@/design-system/loading";

type CameraConfig = z.infer<
  (typeof peripheralApi)["amendCameraConfiguration"]["requestSchema"]
>["cameraConfig"];

function CameraConfiguration() {
  const alertSnackbars = useAlertSnackbars();
  const [
    cameraConfig,
    updateCameraConfig,
    {
      register,
      handleSubmit,
      formState: { errors },
    },
    { handleSaveConfiguration },
  ] = useCameraConfiguration({ alertSnackbars });

  const handleTextFieldChange =
    (
      field: keyof CameraConfig,
    ): React.ChangeEventHandler<HTMLInputElement | HTMLTextAreaElement> =>
      (e) => {
        updateCameraConfig({ [field]: e.target.value });
      };

  if (!cameraConfig) {
    return <Loading title="Retrieving current camera config..." />;
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
          gap: "1rem",
        }}
      >
        <FormField
          fullWidth
          type="number"
          label="Number of frames to capture"
          {...register("cameraConfig.numFramesToCapture", {
            valueAsNumber: true,
          })}
          value={cameraConfig.numFramesToCapture}
          onChange={handleTextFieldChange("numFramesToCapture")}
          errorMessage={errors.cameraConfig?.numFramesToCapture?.message ?? ""}
        />
        <FormField
          fullWidth
          type="number"
          label="Frames per second"
          {...register("cameraConfig.fps", { valueAsNumber: true })}
          value={cameraConfig.fps}
          onChange={handleTextFieldChange("fps")}
          errorMessage={errors.cameraConfig?.fps?.message ?? ""}
        />
        <FormField
          fullWidth
          type="number"
          label="Wait for next frame batch factor"
          {...register("cameraConfig.waitForNextBatchFactor", {
            valueAsNumber: true,
          })}
          value={cameraConfig.waitForNextBatchFactor}
          onChange={handleTextFieldChange("waitForNextBatchFactor")}
          errorMessage={
            errors.cameraConfig?.waitForNextBatchFactor?.message ?? ""
          }
        />
        <Button type="submit">Save Configuration</Button>
      </Container>
    </Form>
  );
}

export default CameraConfiguration;
