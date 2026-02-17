import {
  indexRouter,
  loginRoute,
  rootRouter,
  dashboardRoute,
  registrationRoute,
  uploadRoute,
} from ".";

export const routeTree = rootRouter.addChildren([
  indexRouter,
  loginRoute,
  dashboardRoute,
  registrationRoute,
  uploadRoute,
]);
