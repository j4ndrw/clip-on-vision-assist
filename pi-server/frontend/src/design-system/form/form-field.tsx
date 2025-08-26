import type { TextFieldProps } from "@mui/material/TextField";
import TextField from "@mui/material/TextField";
import Snackbar from "../snackbar";

const FormField: React.FC<TextFieldProps & { errorMessage: string }> = ({
  errorMessage,
  ...props
}) => (
  <>
    <TextField {...props} error={!!errorMessage} />
    <Snackbar
      message={errorMessage}
      severity="error"
      autoHideDuration='none'
    />
  </>
);
export default FormField;
