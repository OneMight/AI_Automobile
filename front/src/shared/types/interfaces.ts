import type React from "react";
import type { DeterminedModel, User } from "./types";

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
  id: number;
  user: User | null;
  isLoading: boolean;
  isError: boolean;
}
export interface StatisticBlockProps {
  title: string;
  value: number | string;
  icon: React.ReactNode;
}
export interface ModelBlockProps {
  model: DeterminedModel;
}
