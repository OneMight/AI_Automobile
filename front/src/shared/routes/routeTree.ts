import {
  indexRouter,
  loginRoute,
  rootRouter,
  dashboardRoute,
  registrationRoute,
  uploadRoute,
  historyPage,
  reviewsPage,
} from ".";

export const routeTree = rootRouter.addChildren([
  indexRouter,
  loginRoute,
  dashboardRoute,
  registrationRoute,
  uploadRoute,
  historyPage,
  reviewsPage,
]);
