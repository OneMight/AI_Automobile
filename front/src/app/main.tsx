import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/index.css";
import { QueryProvider } from "./providers/QueryProvider.tsx";
import { ProviderRouter } from "./providers/RouterProvider.tsx";
import "@/shared/i18n/i18n.ts";
import { UserProvider } from "./providers/UserProvider.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryProvider>
      <UserProvider>
        <ProviderRouter />
      </UserProvider>
    </QueryProvider>
  </StrictMode>,
);
