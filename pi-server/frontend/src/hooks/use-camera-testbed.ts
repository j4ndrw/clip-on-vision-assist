/* eslint-disable react-hooks/exhaustive-deps */
import { isCameraConnected as isCameraConnectedService } from '@/services/testbed/is-camera-connected';
import { useEffect, useRef, useState } from "react";
import type { useAlertSnackbars } from './use-alert-snackbars';

const CHECK_IF_CAMERA_CONNECTED_INTERVAL = 5000;

type Props = {
  alertSnackbars?: ReturnType<typeof useAlertSnackbars>;
};

export const useCameraTestbed = ({ alertSnackbars }: Props) => {
  const isCameraConnectedRequestIds = useRef<string[]>([]);
  const isCameraConnectedInterval = useRef<NodeJS.Timeout | null>(null);

  const [isCameraConnected, setIsCameraConnected] = useState<boolean>();

  const markCameraAsDisconnected = () => {
    setIsCameraConnected(false)
  }

  useEffect(() => {
    isCameraConnectedInterval.current = setInterval(async () => {
      const request = isCameraConnectedService({
        cancelIf: (requestId) =>
          isCameraConnectedRequestIds.current.length > 0 &&
          isCameraConnectedRequestIds.current.at(-1) !== requestId,
        onCancel: (requestId) =>
        (isCameraConnectedRequestIds.current = isCameraConnectedRequestIds.current.filter(
          (id) => id !== requestId,
        )),
        onValidationError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
          setIsCameraConnected(false);
        },
        onApiError: (error) => {
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage(error.message);
          setIsCameraConnected(false);
        },
        onSuccess: ({ isCameraConnected }) => {
          if (isCameraConnectedInterval.current) {
            isCameraConnectedRequestIds.current = [];
          }
          alertSnackbars?.setSnackbarSuccessMessage("");
          alertSnackbars?.setSnackbarErrorMessage("");
          setIsCameraConnected(isCameraConnected)
        },
      });
      isCameraConnectedRequestIds.current.push(request.requestId);
      return request.promise();
    }, CHECK_IF_CAMERA_CONNECTED_INTERVAL);

    return () => {
      if (isCameraConnectedInterval.current)
        clearInterval(isCameraConnectedInterval.current);
    };
  }, []);

  return { isCameraConnected, markCameraAsDisconnected } as const;
};
