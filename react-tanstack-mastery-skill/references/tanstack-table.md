# TanStack Table Guide

## Table of Contents
1. [Basic Setup](#basic-setup)
2. [Column Definition](#column-definition)
3. [Sorting](#sorting)
4. [Filtering](#filtering)
5. [Pagination](#pagination)
6. [Row Selection](#row-selection)
7. [Server-Side Operations](#server-side-operations)
8. [Virtual Scrolling](#virtual-scrolling)

## Basic Setup

```typescript
// features/users/components/UserTable.tsx
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  type ColumnDef,
} from '@tanstack/react-table'
import type { User } from '../types'

interface UserTableProps {
  data: User[]
}

export function UserTable({ data }: UserTableProps) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

  return (
    <table>
      <thead>
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th key={header.id}>
                {flexRender(
                  header.column.columnDef.header,
                  header.getContext()
                )}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
```

## Column Definition

```typescript
import { createColumnHelper } from '@tanstack/react-table'

const columnHelper = createColumnHelper<User>()

// Type-safe column definitions
const columns = [
  // Simple accessor
  columnHelper.accessor('id', {
    header: 'ID',
    cell: (info) => info.getValue(),
    size: 80,
  }),

  // Accessor dengan render custom
  columnHelper.accessor('name', {
    header: 'Name',
    cell: (info) => (
      <div className="font-medium">{info.getValue()}</div>
    ),
    enableSorting: true,
    sortingFn: 'alphanumeric',
  }),

  // Accessor dengan computed value
  columnHelper.accessor(
    (row) => `${row.firstName} ${row.lastName}`,
    {
      id: 'fullName',
      header: 'Full Name',
    }
  ),

  // Accessor row untuk complex cell
  columnHelper.accessor('email', {
    header: 'Email',
    cell: ({ row }) => (
      <a href={`mailto:${row.original.email}`}>
        {row.original.email}
      </a>
    ),
  }),

  // Display column (no accessor)
  columnHelper.display({
    id: 'actions',
    header: () => 'Actions',
    cell: ({ row }) => (
      <RowActions user={row.original} />
    ),
  }),

  // Group header
  columnHelper.group({
    header: 'Contact Info',
    columns: [
      columnHelper.accessor('email', { header: 'Email' }),
      columnHelper.accessor('phone', { header: 'Phone' }),
    ],
  }),
]
```

## Sorting

```typescript
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  type SortingState,
} from '@tanstack/react-table'

function SortableTable({ data }: { data: User[] }) {
  const [sorting, setSorting] = useState<SortingState>([])

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableMultiSort: true, // Shift+Click untuk multi-sort
  })

  return (
    <table>
      <thead>
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th
                key={header.id}
                onClick={header.column.getToggleSortingHandler()}
                className={header.column.getCanSort() ? 'cursor-pointer' : ''}
              >
                {flexRender(
                  header.column.columnDef.header,
                  header.getContext()
                )}
                {/* Sort indicator */}
                {{
                  asc: ' 🔼',
                  desc: ' 🔽',
                }[header.column.getIsSorted() as string] ?? null}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      {/* ... */}
    </table>
  )
}
```

## Filtering

```typescript
import {
  getFilteredRowModel,
  type ColumnFiltersState,
} from '@tanstack/react-table'

function FilterableTable({ data }: { data: User[] }) {
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilter, setGlobalFilter] = useState('')

  const table = useReactTable({
    data,
    columns,
    state: {
      columnFilters,
      globalFilter,
    },
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    // Global filter function
    globalFilterFn: 'includesString',
  })

  return (
    <div>
      {/* Global search */}
      <input
        value={globalFilter ?? ''}
        onChange={(e) => setGlobalFilter(e.target.value)}
        placeholder="Search all columns..."
      />

      {/* Column filter */}
      <input
        value={(table.getColumn('name')?.getFilterValue() as string) ?? ''}
        onChange={(e) =>
          table.getColumn('name')?.setFilterValue(e.target.value)
        }
        placeholder="Filter by name..."
      />

      <table>{/* ... */}</table>
    </div>
  )
}

// Custom filter function
const columns = [
  columnHelper.accessor('status', {
    filterFn: (row, columnId, filterValue) => {
      if (filterValue === 'all') return true
      return row.getValue(columnId) === filterValue
    },
  }),
]
```

## Pagination

```typescript
import { getPaginationRowModel } from '@tanstack/react-table'

function PaginatedTable({ data }: { data: User[] }) {
  const [pagination, setPagination] = useState({
    pageIndex: 0,
    pageSize: 10,
  })

  const table = useReactTable({
    data,
    columns,
    state: { pagination },
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  })

  return (
    <div>
      <table>{/* ... */}</table>

      <div className="pagination">
        <button
          onClick={() => table.firstPage()}
          disabled={!table.getCanPreviousPage()}
        >
          {'<<'}
        </button>
        <button
          onClick={() => table.previousPage()}
          disabled={!table.getCanPreviousPage()}
        >
          {'<'}
        </button>
        <span>
          Page {table.getState().pagination.pageIndex + 1} of{' '}
          {table.getPageCount()}
        </span>
        <button
          onClick={() => table.nextPage()}
          disabled={!table.getCanNextPage()}
        >
          {'>'}
        </button>
        <button
          onClick={() => table.lastPage()}
          disabled={!table.getCanNextPage()}
        >
          {'>>'}
        </button>

        <select
          value={table.getState().pagination.pageSize}
          onChange={(e) => table.setPageSize(Number(e.target.value))}
        >
          {[10, 20, 50, 100].map((size) => (
            <option key={size} value={size}>
              Show {size}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
```

## Row Selection

```typescript
import { type RowSelectionState } from '@tanstack/react-table'

function SelectableTable({ data }: { data: User[] }) {
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({})

  const columns = useMemo(() => [
    // Checkbox column
    {
      id: 'select',
      header: ({ table }) => (
        <input
          type="checkbox"
          checked={table.getIsAllRowsSelected()}
          indeterminate={table.getIsSomeRowsSelected()}
          onChange={table.getToggleAllRowsSelectedHandler()}
        />
      ),
      cell: ({ row }) => (
        <input
          type="checkbox"
          checked={row.getIsSelected()}
          disabled={!row.getCanSelect()}
          onChange={row.getToggleSelectedHandler()}
        />
      ),
    },
    // ...other columns
  ], [])

  const table = useReactTable({
    data,
    columns,
    state: { rowSelection },
    onRowSelectionChange: setRowSelection,
    getCoreRowModel: getCoreRowModel(),
    enableRowSelection: true,
    // Optional: disable selection untuk row tertentu
    enableRowSelection: (row) => row.original.status !== 'deleted',
  })

  // Get selected rows
  const selectedRows = table.getSelectedRowModel().rows
  const selectedUsers = selectedRows.map((row) => row.original)

  return (
    <div>
      <p>{Object.keys(rowSelection).length} rows selected</p>
      <button onClick={() => handleBulkDelete(selectedUsers)}>
        Delete Selected
      </button>
      <table>{/* ... */}</table>
    </div>
  )
}
```

## Server-Side Operations

```typescript
// Sync state dengan URL (TanStack Router)
import { useNavigate, useSearch } from '@tanstack/react-router'

function ServerTable() {
  const search = useSearch({ from: '/users' })
  const navigate = useNavigate({ from: '/users' })

  // Fetch data dengan filters
  const { data, isLoading } = useQuery({
    queryKey: ['users', search],
    queryFn: () => fetchUsers(search),
  })

  const table = useReactTable({
    data: data?.items ?? [],
    columns,
    pageCount: data?.totalPages ?? -1,
    state: {
      pagination: {
        pageIndex: search.page - 1,
        pageSize: search.pageSize,
      },
      sorting: search.sort ? [{ id: search.sort, desc: search.order === 'desc' }] : [],
    },
    onPaginationChange: (updater) => {
      const newState = typeof updater === 'function'
        ? updater({ pageIndex: search.page - 1, pageSize: search.pageSize })
        : updater
      navigate({
        search: (prev) => ({
          ...prev,
          page: newState.pageIndex + 1,
          pageSize: newState.pageSize,
        }),
      })
    },
    onSortingChange: (updater) => {
      const newState = typeof updater === 'function'
        ? updater(search.sort ? [{ id: search.sort, desc: search.order === 'desc' }] : [])
        : updater
      navigate({
        search: (prev) => ({
          ...prev,
          sort: newState[0]?.id,
          order: newState[0]?.desc ? 'desc' : 'asc',
          page: 1, // Reset page on sort change
        }),
      })
    },
    manualPagination: true,
    manualSorting: true,
    getCoreRowModel: getCoreRowModel(),
  })

  return <table>{/* ... */}</table>
}
```

## Virtual Scrolling

```typescript
import { useVirtualizer } from '@tanstack/react-virtual'

function VirtualTable({ data }: { data: User[] }) {
  const tableContainerRef = useRef<HTMLDivElement>(null)

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  })

  const { rows } = table.getRowModel()

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 50, // Estimated row height
    overscan: 10,
  })

  return (
    <div
      ref={tableContainerRef}
      className="h-[600px] overflow-auto"
    >
      <table>
        <thead>{/* ... */}</thead>
        <tbody
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            position: 'relative',
          }}
        >
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const row = rows[virtualRow.index]
            return (
              <tr
                key={row.id}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  transform: `translateY(${virtualRow.start}px)`,
                  height: `${virtualRow.size}px`,
                }}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
```
