import { AlertCircleIcon } from "lucide-react";
import { Alert, AlertDescription } from "./Alert";
import { useEffect, useState } from "react";

interface CustomAlertProps {
  error: string;
  setIsError: (value: string) => void;
}

export const CustomAlert = ({ error, setIsError }: CustomAlertProps) => {
  const [isVisible, setIsVisible] = useState(true);
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setIsError("");
    }, 2000);
    return () => clearTimeout(timer);
  }, [setIsError]);

  if (!isVisible) return null;
  return (
    <div className="grid w-full max-w-75 items-start  gap-4 absolute top-10">
      <Alert variant={"destructive"} className="bg-transparent">
        <AlertCircleIcon />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    </div>
  );
};
