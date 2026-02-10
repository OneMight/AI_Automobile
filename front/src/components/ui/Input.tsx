import * as React from "react";

import { cn } from "@/lib/utils";

function Input({
  className,
  type,
  children,
  ...props
}: React.ComponentProps<"input">) {
  return (
    <div className="relative w-full group">
      <input
        type={type}
        data-slot="input"
        className={cn(
          "px-10",
          "placeholder:text-secondary-text bg-main-app focus:border-main/50 transition-all border-input h-10 w-full min-w-0 rounded-md border py-2 text-base shadow-xs outline-none  disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
          "border-transparent",
          "aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
          className,
        )}
        {...props}
      />
      {children}
    </div>
  );
}

export { Input };
