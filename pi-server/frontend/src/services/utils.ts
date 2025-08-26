/* eslint-disable @typescript-eslint/no-explicit-any */
import type { ApiError } from "@/api/types";
import type { RequestRet } from "@/api/utils";
import type { DeferredPromise } from "@/utils";
import type z from "zod";

const createService =
  <
    TApi extends {
      request: (...args: any[]) => DeferredPromise<any, { requestId: string }>;
    },
    TOptions extends object,
  >(
    requestFactory: (options: TOptions) => ReturnType<TApi["request"]>,
    handler: (
      options: TOptions,
      requestResult: Awaited<
        ReturnType<ReturnType<TApi["request"]>["promise"]>
      >,
    ) => void,
  ) =>
    (options: TOptions) => {
      const request = requestFactory(options);
      return {
        promise: async () => handler(options, await request.promise()),
        requestId: request.requestId,
      };
    };

export const createMutationService = <
  TRequestSchema extends z.ZodType | undefined,
  TRequest extends (...args: any[]) => RequestRet<z.ZodAny>,
  TApi extends
  | { request: TRequest; requestSchema?: undefined }
  | { request: TRequest; requestSchema: TRequestSchema },
  TOptions extends (TApi["requestSchema"] extends TRequestSchema
    ? { input: z.infer<TApi["requestSchema"]> }
    : object) & {
      onApiError: (error: ApiError) => void;
      onSuccess: () => void;
      onCancel?: (requestId: string) => void;
      cancelIf?: (requestId: string) => boolean;
    },
>(
  api: TApi,
) =>
  createService<typeof api, TOptions>(
    (options) =>
      api.request(
        "input" in options ? options.input : undefined,
        options.cancelIf,
      ) as any,
    (options, { requestId, apiError, cancelled }) => {
      if (cancelled) {
        options.onCancel?.(requestId);
        return;
      }

      if (apiError) {
        options.onApiError(apiError);
        return;
      }

      options.onSuccess();
      return;
    },
  );

export const createQueryService = <
  TResponseSchema extends z.ZodType,
  TRequest extends (...args: any[]) => RequestRet<TResponseSchema>,
  TApi extends { request: TRequest; responseSchema: TResponseSchema },
  TOptions extends {
    onValidationError: (
      error: z.ZodError<z.infer<TApi["responseSchema"]>>,
    ) => void;
    onApiError: (error: ApiError) => void;
    onSuccess: (data: z.infer<TApi["responseSchema"]>) => void;
    onCancel?: (requestId: string) => void;
    cancelIf?: (requestId: string) => boolean;
  },
>(
  api: TApi,
) =>
  createService<typeof api, TOptions>(
    (options) => api.request(undefined, options.cancelIf) as any,
    (options, { requestId, data, error, apiError, cancelled }) => {
      if (cancelled) {
        options.onCancel?.(requestId);
        return;
      }

      if (error) {
        options.onValidationError(error);
        return;
      }

      if (apiError) {
        options.onApiError(apiError);
        return;
      }

      options.onSuccess(data);
      return;
    },
  );

export const createBareQueryService = <
  TRequest extends (...args: any[]) => RequestRet<z.ZodAny>,
  TApi extends { request: TRequest },
  TOptions extends {
    onApiError: (error: ApiError) => void;
    onSuccess: () => void;
    onCancel?: (requestId: string) => void;
    cancelIf?: (requestId: string) => boolean;
  },
>(
  api: TApi,
) =>
  createService<typeof api, TOptions>(
    (options) => api.request(undefined, options.cancelIf) as any,
    (options, { requestId, apiError, cancelled }) => {
      if (cancelled) {
        options.onCancel?.(requestId);
        return;
      }

      if (apiError) {
        options.onApiError(apiError);
        return;
      }

      options.onSuccess();
      return;
    },
  );
