import type React from "react";
import type { User } from "./types";

export interface BenefitProps {
  children: React.ReactElement;
  title: string;
  description: string;
}
export interface CustomLinkProps {
  route: string;
  children: React.ReactNode;
  className?: string;
  icon: React.ReactNode;
  isActive: boolean;
}

export interface UserContextType {
  user: User | null;
  isLoading: boolean;
  isError: boolean;
}
