import {
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  getFilteredRowModel,
} from "@tanstack/react-table";
import type {
  ColumnFiltersState,
  ColumnDef,
  SortingState,
} from "@tanstack/react-table";

import { Button, Input, Table } from "@/components";
import { useTranslation } from "react-i18next";
import { useState } from "react";

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
}

export function DataTable<TData, TValue>({
  columns,
  data,
}: DataTableProps<TData, TValue>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const { t } = useTranslation("Table");
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    state: {
      sorting,
      columnFilters,
    },
  });
  return (
    <div className="flex flex-col w-full gap-5">
      <div className="overflow-hidden w-full rounded-md border border-white/30 bg-button-stroke ">
        <Input
          className="w-60 my-3 ml-3 pl-3"
          placeholder={t("search")}
          value={(table.getColumn("mark")?.getFilterValue() as string) ?? ""}
          onChange={(event) =>
            table.getColumn("mark")?.setFilterValue(event.target.value)
          }
        />
        <Table.Table>
          <Table.TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <Table.TableRow key={headerGroup.id} className=" border-white/30">
                {headerGroup.headers.map((header) => {
                  return (
                    <Table.TableHead
                      className="text-white hover:text-main"
                      key={header.id}
                    >
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext(),
                          )}
                    </Table.TableHead>
                  );
                })}
              </Table.TableRow>
            ))}
          </Table.TableHeader>
          <Table.TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <Table.TableRow
                  className="group hover:bg-white/5 border-white/30"
                  key={row.id}
                  data-state={row.getIsSelected() && "selected"}
                >
                  {row.getVisibleCells().map((cell) => (
                    <Table.TableCell
                      key={cell.id}
                      className="text-secondary-text group-hover:text-white py-3"
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </Table.TableCell>
                  ))}
                </Table.TableRow>
              ))
            ) : (
              <Table.TableRow>
                <Table.TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  No results.
                </Table.TableCell>
              </Table.TableRow>
            )}
          </Table.TableBody>
        </Table.Table>
      </div>
      <div className="flex flex-row w-full items-center justify-between mb-10">
        <p>
          {t("page")}: {table.getState().pagination.pageIndex + 1} {t("of")}{" "}
          {table.getPageCount()}
        </p>
        <div className="flex flex-row items-center gap-2">
          <Button
            className="hover:bg-main/70 bg-transparent"
            onClick={table.previousPage}
            disabled={!table.getCanPreviousPage()}
          >
            {t("previos")}
          </Button>
          <Button
            className="hover:bg-main/70 bg-transparent"
            onClick={table.nextPage}
            disabled={!table.getCanNextPage()}
          >
            {t("next")}
          </Button>
        </div>
      </div>
    </div>
  );
}
