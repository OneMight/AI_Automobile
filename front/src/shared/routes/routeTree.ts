import { indexRouter, loginRoute, rootRouter } from ".";

export const routeTree = rootRouter.addChildren([indexRouter, loginRoute]);
