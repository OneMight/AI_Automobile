import App from "@/app/App";
import { createRootRoute, createRoute, redirect } from "@tanstack/react-router";
import { ROUTES } from "./routesPath";
import React from "react";
import { getUserRole } from "@/lib/auth";

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
const uploadRoute = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.UPLOAD,
  component: React.lazy(() =>
    import("@/pages/ui/Upload").then((module) => ({
      default: module.Upload,
    })),
  ),
});
const historyPage = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.HISTORY,
  component: React.lazy(() =>
    import("@/pages/index").then((module) => ({
      default: module.History,
    })),
  ),
});
const reviewsPage = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.REVIEWS,
  component: React.lazy(() =>
    import("@/pages/index").then((module) => ({
      default: module.Reviews,
    })),
  ),
});
const ownerPage = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.OWNER,
  beforeLoad: async () => {
    const user = await getUserRole();
    if (user?.role === null) {
      throw redirect({ to: ROUTES.HOME });
    } else if (user?.role !== "owner") {
      if (user?.role == "admin") {
        throw redirect({ to: ROUTES.ADMIN });
      } else {
        throw redirect({ to: ROUTES.DASHBOARD });
      }
    }
  },
  component: React.lazy(() =>
    import("@/pages/index").then((module) => ({
      default: module.Owner,
    })),
  ),
});
const adminPage = createRoute({
  getParentRoute: () => rootRouter,
  path: ROUTES.ADMIN,
  beforeLoad: async () => {
    const user = await getUserRole();
    if (user?.role === null) {
      throw redirect({ to: ROUTES.HOME });
    } else if (user?.role !== "admin" && user?.role !== "owner") {
      throw redirect({ to: ROUTES.DASHBOARD });
    }
  },
  component: React.lazy(() =>
    import("@/pages/index").then((module) => ({
      default: module.Admin,
    })),
  ),
});
export {
  rootRouter,
  indexRouter,
  loginRoute,
  dashboardRoute,
  registrationRoute,
  uploadRoute,
  historyPage,
  reviewsPage,
  ownerPage,
  adminPage,
};
