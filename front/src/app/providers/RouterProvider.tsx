import { createRouter, RouterProvider } from "@tanstack/react-router";
import { routeTree } from "@/shared/routes/routeTree";

const routes = createRouter({
  routeTree,
});

export const ProviderRouter = () => {
  return <RouterProvider router={routes} />;
};
