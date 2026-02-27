export const ROUTES = {
  HOME: "/",
  LOGIN: "/login",
  REGISTER: "/registration",
  DASHBOARD: "/dashboard",
  UPLOAD: "/upload",
  HISTORY: "/history",
  REVIEWS: "/reviews",
  OWNER: "/owner",
  ADMIN: "/admin",
  SWAGGER: `${import.meta.env.VITE_API_URL}/api-docs`,
} as const;
