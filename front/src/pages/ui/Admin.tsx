import { useUser } from "@/lib/useUser";
import { Reviews } from "./Reviews";
import { isAdmin } from "@/lib/utils";

export const Admin = () => {
  const { user } = useUser();
  const userIsAdmin = isAdmin(user?.role);
  return <Reviews admin={userIsAdmin} />;
};
