import type React from "react";
import type { DeterminedModel, Reviews, User } from "./types";
import type { ReactNode } from "react";

export interface BenefitProps {
  children: React.ReactElement;
  title: string;
  description: string;
}
export interface CustomLinkProps {
  route: string;
  children: React.ReactNode;
  className?: string;
  icon?: React.ReactNode;
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
export interface RecognitionErrorProps {
  title: string;
  desctiption: string | null;
  setError: (value: null) => void;
}
export interface AccordionProps<TData> {
  children: ReactNode;
  content: TData;
}
export interface StarRatingProps {
  rating: number;
  interactive?: boolean;
  onRatingChange?: (rating: number) => void;
  size?: number;
  className?: string;
}
export interface ReviewsBlockProps {
  review: Reviews;
  className?: string;
  admin?: boolean;
}
export interface AgeData {
  category: string;
  count: number;
}

export interface PropsAge {
  data: AgeData[];
}

export interface RecognitionData {
  day: string;
  count: number;
}

export interface PropsLine {
  data: RecognitionData[];
}
export interface ReviewProps {
  admin?: boolean;
}
