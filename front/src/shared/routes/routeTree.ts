import {
  indexRouter,
  loginRoute,
  rootRouter,
  dashboardRoute,
  registrationRoute,
} from ".";

export const routeTree = rootRouter.addChildren([
  indexRouter,
  loginRoute,
  dashboardRoute,
  registrationRoute,
]);
