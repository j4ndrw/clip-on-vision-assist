/* eslint-disable @typescript-eslint/no-explicit-any */
export type DeferredPromise<
  TPromiseRet,
  TExtension extends object,
> = TExtension & {
  promise: () => Promise<TPromiseRet>;
};

export const sleep = async (ms: number) =>
  new Promise((res) => setTimeout(res, ms));

export const debounce = <T extends (...args: any[]) => Promise<any>>(
  fn: T,
  ms: number,
) => {
  let timer: NodeJS.Timeout;

  return ((...args) =>
    new Promise((resolve) => {
      if (timer) {
        clearTimeout(timer);
      }

      timer = setTimeout(() => {
        resolve(fn(...args));
      }, ms);
    })) as T;
};

export const withRetry = <T extends (...args: any[]) => Promise<any>>(
  factory: (triggerRetry: () => void) => T,
  delayMs: number,
): T => {
  let shouldRetry = false;
  const triggerRetry = () => (shouldRetry = true);

  return (async (...args) => {
    let result: ReturnType<ReturnType<typeof factory>>
    while (true) {
      const cb = factory(triggerRetry);
      result = await cb(...args);

      if (!shouldRetry) break;

      await sleep(delayMs)
      shouldRetry = false;
    }
    return result;
  }) as T
};
