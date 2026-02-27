import { SquareCheckBig } from "lucide-react";
import { Alert, AlertDescription } from "./Alert";
import { useEffect, useState } from "react";

interface CustomAlertProps {
  title: string;
  setIsSuccessFeedback: (value: boolean) => void;
}

export const SuccessAlert = ({
  title,
  setIsSuccessFeedback,
}: CustomAlertProps) => {
  const [isVisible, setIsVisible] = useState(true);
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setIsSuccessFeedback(false);
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  if (!isVisible) return null;
  return (
    <div className="grid w-full max-w-75 items-start gap-4 absolute top-10">
      <Alert variant={"success"} className="bg-transparent">
        <SquareCheckBig />
        <AlertDescription>{title}</AlertDescription>
      </Alert>
    </div>
  );
};
