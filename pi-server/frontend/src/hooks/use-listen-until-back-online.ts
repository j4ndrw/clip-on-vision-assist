import { healthCheck } from "@/services/health-check";
import { sleep } from "@/utils";
import { useCallback, useEffect, useRef } from "react";

type Props = {
  preflightDelayMs?: number;
  checkEveryMs: number;
  onSuccess: () => void;
};

export const useListenUntilBackOnline = ({
  preflightDelayMs,
  checkEveryMs,
  onSuccess,
}: Props) => {
  const healthCheckRequestIds = useRef<string[]>([]);
  const healthCheckInterval = useRef<NodeJS.Timeout | null>(null);

  const listenUntilBackOnline = useCallback(async () => {
    if (preflightDelayMs) await sleep(preflightDelayMs);

    healthCheckInterval.current = setInterval(async () => {
      const request = healthCheck({
        cancelIf: (requestId) =>
          healthCheckRequestIds.current.length > 0 &&
          healthCheckRequestIds.current.at(-1) !== requestId,
        onCancel: (requestId) =>
        (healthCheckRequestIds.current = healthCheckRequestIds.current.filter(
          (id) => id !== requestId,
        )),
        onApiError: () => { }, // Can be skipped since we're just listening to see when the connection comes back
        onSuccess: () => {
          if (healthCheckInterval.current) {
            clearInterval(healthCheckInterval.current);
            healthCheckInterval.current = null;
            healthCheckRequestIds.current = [];
          }
          onSuccess();
        },
      });
      healthCheckRequestIds.current.push(request.requestId);
      return request.promise();
    }, checkEveryMs);
  }, [preflightDelayMs, checkEveryMs, onSuccess]);

  useEffect(() => {
    return () => {
      if (healthCheckInterval.current)
        clearInterval(healthCheckInterval.current);
    };
  }, []);

  return { listenUntilBackOnline, healthCheckInterval };
};
