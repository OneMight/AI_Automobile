import type { Role } from "@/shared/types/types";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function isOwner(value: Role | undefined): value is "owner" {
  return value === "owner";
}
export function isAdmin(value: Role | undefined): value is "admin" {
  return value === "admin";
}
