import { cn } from "@/lib/utils";
import type { CustomLinkProps } from "@/shared/types/interfaces";

export const CustomLink = ({ children, icon, isActive }: CustomLinkProps) => {
  return (
    <div
      className={cn(
        "transition-all duration-300 ease-in-out",
        "flex flex-row items-center rounded-xl p-2 gap-2 group-hover:text-main duration-200",
        isActive
          ? "underline underline-offset-5 bg-main/10 border-main text-main hover:text-main/50 hover:border-main/50"
          : "bg-transparent text-secondary-text",
      )}
    >
      {icon}
      {children}
    </div>
  );
};
