import { useReducer } from "react";

export const useReducedState = <T extends object>(initialState: T) => {
  const [state, setState] = useReducer<
    T | null,
    [Partial<T>]
  >(
    (prev, next) => ({
      ...(prev ?? initialState),
      ...next,
    }),
    null,
  );

  return [state, setState] as const
}

export const useNonNullReducedState = <T extends object>(initialState: T) => {
  const [state, setState] = useReducer<
    T,
    [Partial<T>]
  >(
    (prev, next) => ({
      ...prev,
      ...next,
    }),
    initialState,
  );

  return [state, setState] as const
}
