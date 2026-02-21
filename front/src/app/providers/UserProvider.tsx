import { useMemo } from "react";
import type { ReactNode } from "react";
import { useGetDataToken } from "@/api/userApi";
import { UserContext } from "@/lib/useUser";
export const UserProvider = ({ children }: { children: ReactNode }) => {
  const refreshToken = localStorage.getItem("refreshToken");

  const { user, isLoading, isError } = useGetDataToken(refreshToken);

  const value = useMemo(
    () => ({
      id: user?.id ?? 0,
      user: user ?? null,
      isLoading,
      isError,
    }),
    [user, isLoading, isError],
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
};
