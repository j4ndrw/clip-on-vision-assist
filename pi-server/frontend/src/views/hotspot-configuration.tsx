import { useState } from "react";
import z from "zod";
import { hotspotApi } from "@/api";

import Container from "@mui/material/Container";
import Typography from "@mui/material/Typography";
import FormField from "@/design-system/form/form-field";
import Button from "@mui/material/Button";
import { useAlertSnackbars } from "@/hooks/use-alert-snackbars";
import { useHotspotConfiguration } from "@/hooks/use-hotspot-configuration";
import Loading from "@/design-system/loading";
import Form from "@/design-system/form/form";
import Alert from "@mui/material/Alert";
import OutlinedInput from "@mui/material/OutlinedInput";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import {
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
} from "@mui/icons-material";

type HotspotConfig = z.infer<
  (typeof hotspotApi)["amendHotspotConfiguration"]["requestSchema"]
>["hotspotConfig"];

function HotspotConfiguration() {
  const alertSnackbars = useAlertSnackbars();
  const [
    hotspotConfig,
    updateHotspotConfig,
    {
      register,
      handleSubmit,
      formState: { errors },
    },
    { handleSaveConfiguration },
  ] = useHotspotConfiguration({ alertSnackbars });
  const [showPassword, setShowPassword] = useState(false);

  const handleTextFieldChange =
    (
      field: keyof HotspotConfig,
      options: { sideEffect?: (value: string) => void } = {},
    ): React.ChangeEventHandler<HTMLInputElement | HTMLTextAreaElement> =>
      (e) => {
        updateHotspotConfig({ [field]: e.target.value });
        options.sideEffect?.(e.target.value);
      };

  const handleClickShowPassword = () => {
    setShowPassword((prev) => !prev);
  };

  if (!hotspotConfig) {
    return <Loading title="Retrieving current hotspot config..." />;
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
        <Typography variant="subtitle1">Hotspot Configuration</Typography>
        <Alert variant="outlined" severity="info">
          NOTE: You will have to restart your vision aid satellite after you
          make changes related to your hotspot.
        </Alert>
        <FormField
          fullWidth
          type="text"
          label="Hotspot SSID"
          {...register("hotspotConfig.ssid")}
          value={hotspotConfig.ssid}
          onChange={handleTextFieldChange("ssid")}
          errorMessage={errors.hotspotConfig?.ssid?.message ?? ""}
        />
        <FormField
          textFieldComponent={OutlinedInput}
          fullWidth
          type={showPassword ? "text" : "password"}
          notched
          placeholder="Hotspot Password"
          {...register("hotspotConfig.password")}
          value={hotspotConfig.password}
          onChange={handleTextFieldChange("password")}
          errorMessage={errors.hotspotConfig?.password?.message ?? ""}
          endAdornment={
            <InputAdornment position="end">
              <IconButton
                size="small"
                onClick={handleClickShowPassword}
                edge="end"
              >
                {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
              </IconButton>
            </InputAdornment>
          }
        />
        <Button type="submit">Save Configuration</Button>
      </Container>
    </Form>
  );
}

export default HotspotConfiguration;
