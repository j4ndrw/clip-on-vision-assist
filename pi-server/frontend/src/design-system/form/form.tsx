import Snackbar from "../snackbar";
import type { useAlertSnackbars } from "@/hooks/use-alert-snackbars";

type Props = {
  onSubmit?: React.ComponentProps<"form">["onSubmit"];
  alertSnackbars?: ReturnType<typeof useAlertSnackbars>;
};

const Form = ({
  alertSnackbars,
  onSubmit,
  children,
}: React.PropsWithChildren<Props>) => (
  <>
    <form style={{ width: "100%" }} onSubmit={onSubmit}>
      {children}
    </form>
    {alertSnackbars && (
      <>
        <Snackbar
          message={alertSnackbars.snackbarSuccessMessage}
          severity="success"
          onClose={alertSnackbars.handleSuccessSnackbarClose}
        />
        <Snackbar
          message={alertSnackbars.snackbarErrorMessage}
          severity="error"
          onClose={alertSnackbars.handleErrorSnackbarClose}
        />
      </>
    )}
  </>
);

export default Form;
