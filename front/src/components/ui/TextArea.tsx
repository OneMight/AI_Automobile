import * as React from "react";

import { cn } from "@/lib/utils";

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "focus:border-main/50 disabled:bg-input/50 dark:disabled:bg-input/80 rounded-lg border border-main-app bg-main-app px-2.5 py-2 text-base transition-colors md:text-sm placeholder:text-muted-foreground flex field-sizing-content min-h-40 max-h-40 w-full outline-none disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}

export { Textarea };
