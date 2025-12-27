# TanStack Router Guide

## Table of Contents
1. [Setup](#setup)
2. [Route Definition](#route-definition)
3. [Type-Safe Navigation](#type-safe-navigation)
4. [Search Params (URL State)](#search-params)
5. [Route Loaders](#route-loaders)
6. [Protected Routes](#protected-routes)
7. [Code Splitting](#code-splitting)
8. [Error Handling](#error-handling)

## Setup

```typescript
// app/router.tsx
import { createRouter } from '@tanstack/react-router'
import { QueryClient } from '@tanstack/react-query'
import { routeTree } from './routeTree.gen' // Generated file

export const queryClient = new QueryClient()

export const router = createRouter({
  routeTree,
  context: {
    queryClient,
    auth: undefined!, // Will be set by AuthProvider
  },
  defaultPreload: 'intent', // Preload on hover/focus
  defaultPreloadStaleTime: 0,
})

// Type registration
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

// main.tsx
import { RouterProvider } from '@tanstack/react-router'
import { QueryClientProvider } from '@tanstack/react-query'
import { router, queryClient } from './router'

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  )
}
```

## Route Definition

### Root Route
```typescript
// routes/__root.tsx
import { createRootRouteWithContext, Outlet } from '@tanstack/react-router'
import { QueryClient } from '@tanstack/react-query'
import { TanStackRouterDevtools } from '@tanstack/router-devtools'

interface RouterContext {
  queryClient: QueryClient
  auth: AuthContext | undefined
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
})

function RootLayout() {
  return (
    <>
      <Header />
      <main>
        <Outlet />
      </main>
      <Footer />
      <TanStackRouterDevtools position="bottom-right" />
    </>
  )
}
```

### Static Route
```typescript
// routes/about.tsx
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/about')({
  component: AboutPage,
})

function AboutPage() {
  return <h1>About Us</h1>
}
```

### Dynamic Route
```typescript
// routes/users/$userId.tsx
import { createFileRoute } from '@tanstack/react-router'
import { useSuspenseQuery } from '@tanstack/react-query'
import { userQueries } from '@/features/users/api'

export const Route = createFileRoute('/users/$userId')({
  // Loader untuk prefetch data
  loader: ({ context, params }) => {
    return context.queryClient.ensureQueryData(
      userQueries.detail(params.userId)
    )
  },
  component: UserDetailPage,
})

function UserDetailPage() {
  // Type-safe params
  const { userId } = Route.useParams()
  const { data: user } = useSuspenseQuery(userQueries.detail(userId))

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  )
}
```

### Layout Route
```typescript
// routes/dashboard/_layout.tsx
import { createFileRoute, Outlet } from '@tanstack/react-router'

export const Route = createFileRoute('/dashboard/_layout')({
  component: DashboardLayout,
})

function DashboardLayout() {
  return (
    <div className="dashboard">
      <DashboardSidebar />
      <div className="dashboard-content">
        <Outlet />
      </div>
    </div>
  )
}

// routes/dashboard/_layout/index.tsx → /dashboard
// routes/dashboard/_layout/settings.tsx → /dashboard/settings
```

## Type-Safe Navigation

```typescript
import { Link, useNavigate } from '@tanstack/react-router'

// Link component - type-safe
function Navigation() {
  return (
    <nav>
      {/* Static route */}
      <Link to="/">Home</Link>

      {/* Route with params */}
      <Link 
        to="/users/$userId" 
        params={{ userId: '123' }}
      >
        User Profile
      </Link>

      {/* Route with search params */}
      <Link 
        to="/users" 
        search={{ status: 'active', page: 1 }}
      >
        Active Users
      </Link>

      {/* Active state styling */}
      <Link
        to="/dashboard"
        activeProps={{ className: 'active' }}
        activeOptions={{ exact: true }}
      >
        Dashboard
      </Link>
    </nav>
  )
}

// Programmatic navigation
function UserCard({ userId }: { userId: string }) {
  const navigate = useNavigate()

  const handleClick = () => {
    navigate({
      to: '/users/$userId',
      params: { userId },
      search: { tab: 'profile' },
    })
  }

  const goBack = () => navigate({ to: '..' })

  return (
    <button onClick={handleClick}>View User</button>
  )
}
```

## Search Params

**URL state > React state untuk shareable/bookmarkable state:**

```typescript
// routes/users/index.tsx
import { createFileRoute } from '@tanstack/react-router'
import { z } from 'zod'

// Define search params schema
const userSearchSchema = z.object({
  q: z.string().optional(),
  status: z.enum(['active', 'inactive', 'all']).default('all'),
  page: z.number().min(1).default(1),
  sort: z.enum(['name', 'createdAt']).default('createdAt'),
  order: z.enum(['asc', 'desc']).default('desc'),
})

type UserSearch = z.infer<typeof userSearchSchema>

export const Route = createFileRoute('/users')({
  // Validate search params
  validateSearch: (search) => userSearchSchema.parse(search),
  
  // Loader uses search params
  loaderDeps: ({ search }) => ({ search }),
  loader: ({ context, deps }) => {
    return context.queryClient.ensureQueryData(
      userQueries.list(deps.search)
    )
  },
  component: UsersPage,
})

function UsersPage() {
  // Type-safe search params
  const { q, status, page, sort, order } = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })

  // Update search params
  const setFilter = (key: keyof UserSearch, value: unknown) => {
    navigate({
      search: (prev) => ({
        ...prev,
        [key]: value,
        // Reset page when filter changes
        page: key !== 'page' ? 1 : value,
      }),
    })
  }

  return (
    <div>
      <input
        value={q ?? ''}
        onChange={(e) => setFilter('q', e.target.value)}
        placeholder="Search users..."
      />

      <select
        value={status}
        onChange={(e) => setFilter('status', e.target.value)}
      >
        <option value="all">All</option>
        <option value="active">Active</option>
        <option value="inactive">Inactive</option>
      </select>

      <UserList filters={{ q, status, page, sort, order }} />

      <Pagination
        currentPage={page}
        onPageChange={(p) => setFilter('page', p)}
      />
    </div>
  )
}
```

## Route Loaders

```typescript
// Loader dengan TanStack Query integration
export const Route = createFileRoute('/dashboard')({
  // Dependencies yang trigger re-load
  loaderDeps: ({ search }) => ({ search }),

  // Loader function
  loader: async ({ context, deps, abortController }) => {
    // Parallel data loading
    const [users, stats] = await Promise.all([
      context.queryClient.ensureQueryData(userQueries.list(deps.search)),
      context.queryClient.ensureQueryData(dashboardQueries.stats()),
    ])

    return { users, stats }
  },

  // Loading UI
  pendingComponent: () => <DashboardSkeleton />,

  // Error UI
  errorComponent: ({ error }) => <DashboardError error={error} />,

  component: DashboardPage,
})

function DashboardPage() {
  // Access loader data
  const { users, stats } = Route.useLoaderData()

  // Or use query (recommended untuk reactivity)
  const { data: users } = useSuspenseQuery(userQueries.list())

  return (
    <div>
      <StatsCards stats={stats} />
      <UserTable users={users} />
    </div>
  )
}
```

## Protected Routes

```typescript
// routes/_authenticated.tsx (Layout route)
import { createFileRoute, redirect, Outlet } from '@tanstack/react-router'

export const Route = createFileRoute('/_authenticated')({
  beforeLoad: async ({ context, location }) => {
    // Check auth before loading route
    if (!context.auth?.user) {
      throw redirect({
        to: '/login',
        search: {
          redirect: location.href,
        },
      })
    }
  },
  component: AuthenticatedLayout,
})

function AuthenticatedLayout() {
  return (
    <div className="authenticated-layout">
      <AppSidebar />
      <Outlet />
    </div>
  )
}

// All routes under /_authenticated/ are protected:
// routes/_authenticated/dashboard.tsx
// routes/_authenticated/settings.tsx
// routes/_authenticated/users/$userId.tsx
```

### Role-Based Access

```typescript
// routes/_authenticated/_admin.tsx
export const Route = createFileRoute('/_authenticated/_admin')({
  beforeLoad: async ({ context }) => {
    if (context.auth?.user?.role !== 'admin') {
      throw redirect({ to: '/unauthorized' })
    }
  },
  component: AdminLayout,
})

// Only admins can access:
// routes/_authenticated/_admin/users.tsx
// routes/_authenticated/_admin/settings.tsx
```

## Code Splitting

```typescript
// Lazy load route component
import { createFileRoute } from '@tanstack/react-router'
import { lazy } from 'react'

const HeavyComponent = lazy(() => import('./HeavyComponent'))

export const Route = createFileRoute('/heavy')({
  component: () => (
    <Suspense fallback={<Loading />}>
      <HeavyComponent />
    </Suspense>
  ),
})

// Or use route-level lazy loading
export const Route = createFileRoute('/reports')({
  component: lazyRouteComponent(() => import('./ReportsPage')),
})
```

## Error Handling

```typescript
// Global error boundary di root
export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
  errorComponent: GlobalErrorBoundary,
  notFoundComponent: NotFoundPage,
})

function GlobalErrorBoundary({ error }: { error: Error }) {
  return (
    <div className="error-page">
      <h1>Something went wrong</h1>
      <pre>{error.message}</pre>
      <Link to="/">Go Home</Link>
    </div>
  )
}

// Route-specific error
export const Route = createFileRoute('/users/$userId')({
  loader: async ({ params, context }) => {
    const user = await context.queryClient.ensureQueryData(
      userQueries.detail(params.userId)
    )
    
    if (!user) {
      throw notFound()
    }
    
    return user
  },
  notFoundComponent: () => <UserNotFound />,
  errorComponent: ({ error }) => <UserError error={error} />,
})
```

## Navigation Guards

```typescript
// Confirm before leaving unsaved form
export const Route = createFileRoute('/edit/$id')({
  component: EditPage,
})

function EditPage() {
  const [isDirty, setIsDirty] = useState(false)

  // Block navigation if form is dirty
  useBlocker({
    blockerFn: () => isDirty,
    onBlock: ({ proceed, reset }) => {
      if (window.confirm('Discard unsaved changes?')) {
        proceed()
      } else {
        reset()
      }
    },
  })

  return <Form onDirtyChange={setIsDirty} />
}
```
