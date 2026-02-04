import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/index.css";
import { QueryProvider } from "./providers/QueryProvider.tsx";
import { ProviderRouter } from "./providers/RouterProvider.tsx";
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryProvider>
      <ProviderRouter />
    </QueryProvider>
  </StrictMode>,
);
