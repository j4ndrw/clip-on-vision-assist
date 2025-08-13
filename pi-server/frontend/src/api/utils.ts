import z from "zod";
import { v4 as uuidv4 } from "uuid";
import type { ApiError, Method } from "./types";
import type { DeferredPromise } from "@/utils";

export type ApiRequestConfig<
  TMethod extends Method,
  TRequestSchema extends z.ZodType,
  TResponseSchema extends z.ZodType,
> = {
  endpoint: string;
  requestSchema?: TRequestSchema;
  responseSchema?: TResponseSchema;
  method: TMethod;
  mock?: TResponseSchema extends undefined
  ? undefined
  : z.infer<TResponseSchema>;
};

export type RequestRet<TResponseSchema extends z.ZodType> = DeferredPromise<
  (
    | (z.ZodSafeParseSuccess<z.output<TResponseSchema>> & { apiError: null })
    | (z.ZodSafeParseError<z.output<TResponseSchema>> & { apiError: null })
    | {
      data: undefined;
      error: undefined;
      success: false;
      apiError: ApiError;
    }
  ) & { cancelled?: boolean; requestId: string },
  { requestId: string }
>;

export const createApiRequestConfig = <
  TMethod extends Method,
  TRequestSchema extends z.ZodType,
  TResponseSchema extends z.ZodType,
>(
  config: ApiRequestConfig<TMethod, TRequestSchema, TResponseSchema>,
): ApiRequestConfig<TMethod, TRequestSchema, TResponseSchema> => config;

export const createApiRequest =
  <
    TMethod extends Method,
    TRequestSchema extends z.ZodType,
    TResponseSchema extends z.ZodType,
  >({
    method,
    endpoint,
    requestSchema,
    responseSchema,
    mock,
  }: ApiRequestConfig<TMethod, TRequestSchema, TResponseSchema>): ((
    body?: typeof requestSchema extends undefined
      ? object
      : z.output<TRequestSchema>,
    cancelIf?: (requestId: string) => boolean,
  ) => RequestRet<TResponseSchema>) =>
    (body, cancelIf) => {
      const requestId = uuidv4();
      return {
        requestId,
        promise: async () => {
          const abortController = new AbortController();
          if (import.meta.env.DEV) {
            if (!responseSchema)
              return Object.assign(
                z
                  .any()
                  .nullish()
                  .safeParse(mock ?? {}),
                {
                  requestId,
                  apiError: null,
                },
              );
            return Object.assign(responseSchema.safeParse(mock), {
              requestId,
              apiError: null,
            });
          }

          const requestOptions: RequestInit = {
            method,
            headers: { "Content-Type": "application/json" },
            signal: abortController.signal,
          };
          if (method !== "GET" && !!body)
            requestOptions.body = JSON.stringify(
              requestSchema ? requestSchema.parse(body) : body,
            );

          const cancelCheckInterval = setInterval(() => {
            if (!cancelIf?.(requestId)) return;
            abortController.abort();
            clearInterval(cancelCheckInterval);
          });

          const response = await fetch(endpoint, requestOptions);

          if (!response.ok) {
            const errorJson = await response.json();
            const cancelled = abortController.signal.aborted;

            return {
              requestId,
              cancelled,
              data: undefined,
              error: undefined,
              success: false,
              apiError: {
                message:
                  "message" in errorJson
                    ? (errorJson.message as string)
                    : "Unexpected error - Something went wrong...",
              },
            };
          }

          if (!responseSchema)
            return Object.assign(z.any().safeParse({}), {
              requestId,
              apiError: null,
            });

          const json = await response.json();
          return Object.assign(responseSchema.safeParse(json), {
            requestId,
            apiError: null,
          });
        },
      };
    };
