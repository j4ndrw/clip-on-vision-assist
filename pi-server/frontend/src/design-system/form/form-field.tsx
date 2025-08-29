/* eslint-disable @typescript-eslint/no-explicit-any */
import TextField from "@mui/material/TextField";
import Snackbar from "../snackbar";

type Props<TComponent extends () => React.JSX.Element> = React.ComponentProps<TComponent> & {
  errorMessage: string
  textFieldComponent: TComponent
}

function FormField<TComponent extends (...args: any[]) => React.JSX.Element>({
  errorMessage,
  textFieldComponent: Component = TextField,
  ...props
}: Props<TComponent>) {
  return <>
    <Component {...props} error={!!errorMessage} />
    <Snackbar
      message={errorMessage}
      severity="error"
      autoHideDuration='none'
    />
  </>
};
export default FormField;
