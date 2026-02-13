import App from "@/app/App";
import { createRootRoute, createRoute } from "@tanstack/react-router";
import { ROUTES } from "./routesPath";
import React from "react";

const rootRouter = createRootRoute({
  component: () => <App />,
});
const indexRouter = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.HOME,
  component: React.lazy(() =>
    import("@/pages/index").then((module) => ({
      default: module.Home,
    })),
  ),
});
const loginRoute = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.LOGIN,
  component: React.lazy(() =>
    import("@/pages/index").then((module) => ({
      default: module.Login,
    })),
  ),
});
const dashboardRoute = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.DASHBOARD,
  component: React.lazy(() =>
    import("@/pages/index").then((module) => ({
      default: module.Dashboard,
    })),
  ),
});
const registrationRoute = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.REGISTER,
  component: React.lazy(() =>
    import("@/pages/index").then((module) => ({
      default: module.Registration,
    })),
  ),
});
export {
  rootRouter,
  indexRouter,
  loginRoute,
  dashboardRoute,
  registrationRoute,
};
