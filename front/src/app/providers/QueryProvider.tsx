import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const quertyClient = new QueryClient({
  defaultOptions: {
    queries: {
      gcTime: 1000 * 60 * 60,
    },
  },
});
export type QueryProps = {
  children: ReactNode;
};

export const QueryProvider = ({ children }: QueryProps) => {
  return (
    <QueryClientProvider client={quertyClient}>{children}</QueryClientProvider>
  );
};
