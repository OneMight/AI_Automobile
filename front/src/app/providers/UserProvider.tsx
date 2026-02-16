import { useMemo } from "react";
import type { ReactNode } from "react";
import { useGetDataToken } from "@/api/userApi";
import { UserContext } from "@/lib/useUser";
export const UserProvider = ({ children }: { children: ReactNode }) => {
  const { user, isLoading, isError } = useGetDataToken();

  const value = useMemo(
    () => ({
      user: user ?? null,
      isLoading,
      isError,
    }),
    [user, isLoading, isError],
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
};
