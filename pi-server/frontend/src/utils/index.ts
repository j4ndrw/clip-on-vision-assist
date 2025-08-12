/* eslint-disable @typescript-eslint/no-explicit-any */
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
}
