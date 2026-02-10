import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/index.css";
import { QueryProvider } from "./providers/QueryProvider.tsx";
import { ProviderRouter } from "./providers/RouterProvider.tsx";
import "@/shared/i18n/i18n.ts";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryProvider>
      <ProviderRouter />
    </QueryProvider>
  </StrictMode>,
);
