import { convertDate } from "@/lib/converDate";
import type { ModelTable } from "@/shared/types/types";
import type { ColumnDef } from "@tanstack/react-table";
import { ArrowUpDown } from "lucide-react";
import { Button } from "./Button";
export const columns: ColumnDef<ModelTable>[] = [
  {
    accessorKey: "mark",
    header: ({ column }) => {
      return (
        <Button
          className="hover:bg-transparent bg-transparent hover:text-main"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Mark
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      );
    },
  },
  { accessorKey: "model", header: "Model" },
  { accessorKey: "manufactureYear", header: "Manufacture year" },
  {
    accessorKey: "confidence",
    header: ({ column }) => (
      <div className="w-full flex items-center justify-center">
        <Button
          className="hover:bg-transparent bg-transparent hover:text-main"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Confidence
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      </div>
    ),
    cell: ({ row }) => {
      const data = row.getValue<number>("confidence");
      const formatted = `${data}%`;

      return <div className="text-center">{formatted}</div>;
    },
  },
  {
    accessorKey: "determinedTime",
    header: ({ column }) => (
      <div className="w-full flex items-center justify-center">
        <Button
          className="hover:bg-transparent bg-transparent hover:text-main"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Recognition time
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      </div>
    ),
    cell: ({ row }) => {
      const data = row.getValue<number>("determinedTime");
      const formatted = `${data}s`;

      return <div className="text-center">{formatted}</div>;
    },
  },
  {
    accessorKey: "createdAt",
    header: ({ column }) => (
      <div className="w-full flex items-center justify-end">
        <Button
          className="hover:bg-transparent bg-transparent hover:text-main"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Created At
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      </div>
    ),
    cell: ({ row }) => {
      const date = row.getValue<Date>("createdAt");
      const formatted = convertDate(date);

      return <div className="text-right font-medium">{formatted}</div>;
    },
  },
];
