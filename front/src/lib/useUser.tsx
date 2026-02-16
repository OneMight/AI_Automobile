import type { UserContextType } from "@/shared/types/interfaces";
import { createContext, useContext } from "react";
export const UserContext = createContext<UserContextType>(null!);
export const useUser = () => {
  const context = useContext(UserContext);
  if (context === null) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return context;
};
