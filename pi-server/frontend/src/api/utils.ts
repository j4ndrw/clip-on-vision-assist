import z from "zod";
import type { ApiError, Method } from "./types";

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
  ) => Promise<
    | (z.ZodSafeParseSuccess<z.output<TResponseSchema>> & { apiError: null })
    | (z.ZodSafeParseError<z.output<TResponseSchema>> & { apiError: null })
    | { data: undefined; error: undefined; success: false; apiError: ApiError }
  >) =>
    async (body) => {
      if (import.meta.env.DEV) {
        if (!responseSchema)
          return Object.assign(z.any().nullish().safeParse(mock ?? {}), {
            apiError: null,
          });
        return Object.assign(responseSchema.safeParse(mock), { apiError: null });
      }

      const requestOptions: RequestInit = {
        method,
        headers: { "Content-Type": "application/json" },
      };
      if (method !== "GET" && !!body)
        requestOptions.body = JSON.stringify(requestSchema ? requestSchema.parse(body) : body);

      const response = await fetch(endpoint, requestOptions);
      if (!response.ok) {
        const errorJson = await response.json();

        return {
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
          apiError: null,
        });

      const json = await response.json();
      return Object.assign(responseSchema.safeParse(json), {
        apiError: null,
      });
    };
