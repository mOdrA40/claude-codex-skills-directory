# TanStack Query (React Query) Guide

## Table of Contents
1. [Query Client Setup](#query-client-setup)
2. [Query Keys Factory](#query-keys-factory)
3. [Query Patterns](#query-patterns)
4. [Mutation Patterns](#mutation-patterns)
5. [Optimistic Updates](#optimistic-updates)
6. [Error Handling](#error-handling)
7. [Performance Patterns](#performance-patterns)
8. [Testing Queries](#testing-queries)

## Query Client Setup

```typescript
// lib/query-client.ts
import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Stale time: data considered fresh for 5 minutes
      staleTime: 5 * 60 * 1000,
      // Garbage collection: unused data removed after 30 minutes  
      gcTime: 30 * 60 * 1000,
      // Retry failed requests 3 times with exponential backoff
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      // Refetch on window focus for fresh data
      refetchOnWindowFocus: true,
      // Don't refetch on mount if data is fresh
      refetchOnMount: false,
    },
    mutations: {
      // Show error toast on mutation failure
      onError: (error) => {
        console.error('Mutation failed:', error)
        // toast.error(error.message)
      },
    },
  },
})

// app/providers.tsx
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}
```

## Query Keys Factory

**WAJIB gunakan factory pattern untuk query keys:**

```typescript
// features/users/api/users.queries.ts
import { queryOptions } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import type { User, UserFilter } from '../types'

// Query key factory - SINGLE SOURCE OF TRUTH
export const userKeys = {
  all: ['users'] as const,
  lists: () => [...userKeys.all, 'list'] as const,
  list: (filters: UserFilter) => [...userKeys.lists(), filters] as const,
  details: () => [...userKeys.all, 'detail'] as const,
  detail: (id: string) => [...userKeys.details(), id] as const,
}

// Query options factory - REUSABLE
export const userQueries = {
  list: (filters: UserFilter = {}) =>
    queryOptions({
      queryKey: userKeys.list(filters),
      queryFn: () => api.get<User[]>('/users', { params: filters }),
      staleTime: 5 * 60 * 1000,
    }),

  detail: (id: string) =>
    queryOptions({
      queryKey: userKeys.detail(id),
      queryFn: () => api.get<User>(`/users/${id}`),
      staleTime: 10 * 60 * 1000,
    }),

  // Infinite query untuk pagination
  infinite: (filters: UserFilter = {}) => ({
    queryKey: [...userKeys.list(filters), 'infinite'],
    queryFn: ({ pageParam = 1 }) =>
      api.get<{ data: User[]; nextPage: number | null }>('/users', {
        params: { ...filters, page: pageParam },
      }),
    getNextPageParam: (lastPage) => lastPage.nextPage,
    initialPageParam: 1,
  }),
}
```

## Query Patterns

### Basic Query
```typescript
import { useQuery, useSuspenseQuery } from '@tanstack/react-query'
import { userQueries } from '@/features/users/api'

// Standard query dengan loading state
function UserList() {
  const { data, isLoading, error } = useQuery(userQueries.list())

  if (isLoading) return <Skeleton />
  if (error) return <ErrorMessage error={error} />

  return <ul>{data?.map(user => <UserItem key={user.id} user={user} />)}</ul>
}

// Suspense query (RECOMMENDED dengan React Suspense)
function UserListSuspense() {
  const { data } = useSuspenseQuery(userQueries.list())
  // No loading check needed - Suspense handles it
  return <ul>{data.map(user => <UserItem key={user.id} user={user} />)}</ul>
}

// Parent component
function UsersPage() {
  return (
    <ErrorBoundary fallback={<ErrorMessage />}>
      <Suspense fallback={<Skeleton />}>
        <UserListSuspense />
      </Suspense>
    </ErrorBoundary>
  )
}
```

### Dependent Queries
```typescript
function UserProfile({ userId }: { userId: string }) {
  // First query
  const { data: user } = useSuspenseQuery(userQueries.detail(userId))

  // Second query depends on first
  const { data: posts } = useSuspenseQuery({
    ...postQueries.byUser(user.id),
    // Only fetch if user exists
    enabled: !!user,
  })

  return (
    <div>
      <h1>{user.name}</h1>
      <PostList posts={posts} />
    </div>
  )
}
```

### Parallel Queries
```typescript
import { useSuspenseQueries } from '@tanstack/react-query'

function Dashboard() {
  const [users, stats, notifications] = useSuspenseQueries({
    queries: [
      userQueries.list(),
      dashboardQueries.stats(),
      notificationQueries.unread(),
    ],
  })

  return (
    <div>
      <Stats data={stats.data} />
      <UserList users={users.data} />
      <Notifications items={notifications.data} />
    </div>
  )
}
```

## Mutation Patterns

```typescript
// features/users/api/users.mutations.ts
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api-client'
import { userKeys } from './users.queries'
import type { User, CreateUserDto, UpdateUserDto } from '../types'

export function useCreateUser() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: CreateUserDto) => api.post<User>('/users', data),
    onSuccess: (newUser) => {
      // Invalidate list to refetch
      queryClient.invalidateQueries({ queryKey: userKeys.lists() })
      
      // Optionally set the new user in cache
      queryClient.setQueryData(userKeys.detail(newUser.id), newUser)
    },
  })
}

export function useUpdateUser() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: UpdateUserDto }) =>
      api.patch<User>(`/users/${id}`, data),
    onSuccess: (updatedUser, { id }) => {
      // Update specific user in cache
      queryClient.setQueryData(userKeys.detail(id), updatedUser)
      
      // Invalidate lists (might affect sorting/filtering)
      queryClient.invalidateQueries({ queryKey: userKeys.lists() })
    },
  })
}

export function useDeleteUser() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => api.delete(`/users/${id}`),
    onSuccess: (_, id) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: userKeys.detail(id) })
      
      // Invalidate lists
      queryClient.invalidateQueries({ queryKey: userKeys.lists() })
    },
  })
}
```

## Optimistic Updates

```typescript
export function useUpdateUserOptimistic() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: UpdateUserDto }) =>
      api.patch<User>(`/users/${id}`, data),

    // Optimistic update
    onMutate: async ({ id, data }) => {
      // Cancel in-flight queries
      await queryClient.cancelQueries({ queryKey: userKeys.detail(id) })

      // Snapshot previous value
      const previousUser = queryClient.getQueryData<User>(userKeys.detail(id))

      // Optimistically update
      if (previousUser) {
        queryClient.setQueryData(userKeys.detail(id), {
          ...previousUser,
          ...data,
        })
      }

      // Return context for rollback
      return { previousUser }
    },

    // Rollback on error
    onError: (err, { id }, context) => {
      if (context?.previousUser) {
        queryClient.setQueryData(userKeys.detail(id), context.previousUser)
      }
    },

    // Refetch after mutation
    onSettled: (_, __, { id }) => {
      queryClient.invalidateQueries({ queryKey: userKeys.detail(id) })
    },
  })
}
```

## Error Handling

```typescript
// Global error handler
import { QueryCache, MutationCache } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error, query) => {
      // Only show toast for queries that have already loaded
      if (query.state.data !== undefined) {
        toast.error(`Background refresh failed: ${error.message}`)
      }
    },
  }),
  mutationCache: new MutationCache({
    onError: (error) => {
      toast.error(`Operation failed: ${error.message}`)
    },
  }),
})

// Per-query error handling
const { data, error, isError } = useQuery({
  ...userQueries.detail(userId),
  throwOnError: false, // Handle error in component
})

if (isError) {
  if (error instanceof NotFoundError) {
    return <UserNotFound />
  }
  return <GenericError error={error} />
}

// Type-safe error
import { AxiosError } from 'axios'

interface ApiError {
  message: string
  code: string
}

const { error } = useQuery<User, AxiosError<ApiError>>({
  queryKey: userKeys.detail(userId),
  queryFn: () => api.get(`/users/${userId}`),
})

if (error?.response?.data.code === 'USER_NOT_FOUND') {
  // Handle specific error
}
```

## Performance Patterns

### Prefetching
```typescript
// Prefetch on hover
function UserLink({ userId }: { userId: string }) {
  const queryClient = useQueryClient()

  const prefetch = () => {
    queryClient.prefetchQuery(userQueries.detail(userId))
  }

  return (
    <Link 
      to={`/users/${userId}`}
      onMouseEnter={prefetch}
      onFocus={prefetch}
    >
      View User
    </Link>
  )
}

// Prefetch in route loader (TanStack Router)
export const Route = createFileRoute('/users/$userId')({
  loader: ({ context, params }) => {
    context.queryClient.ensureQueryData(userQueries.detail(params.userId))
  },
  component: UserPage,
})
```

### Pagination
```typescript
import { useInfiniteQuery } from '@tanstack/react-query'

function UserListInfinite() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery(userQueries.infinite())

  return (
    <div>
      {data.pages.map((page) =>
        page.data.map((user) => <UserCard key={user.id} user={user} />)
      )}

      {hasNextPage && (
        <button 
          onClick={() => fetchNextPage()}
          disabled={isFetchingNextPage}
        >
          {isFetchingNextPage ? 'Loading...' : 'Load More'}
        </button>
      )}
    </div>
  )
}
```

### Select for derived data
```typescript
// Select untuk transform data tanpa re-render
const { data: activeUsers } = useQuery({
  ...userQueries.list(),
  select: (users) => users.filter(u => u.isActive),
})

// Dengan stable reference
import { useCallback } from 'react'

const selectActiveUsers = useCallback(
  (users: User[]) => users.filter(u => u.isActive),
  []
)

const { data } = useQuery({
  ...userQueries.list(),
  select: selectActiveUsers,
})
```

## Testing Queries

```typescript
// test-utils.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  })
}

export function renderWithQuery(ui: React.ReactElement) {
  const testQueryClient = createTestQueryClient()
  return render(
    <QueryClientProvider client={testQueryClient}>
      {ui}
    </QueryClientProvider>
  )
}

// UserList.test.tsx
import { screen, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { server } from '@/mocks/server'
import { renderWithQuery } from '@/test-utils'
import { UserList } from './UserList'

describe('UserList', () => {
  it('renders users', async () => {
    server.use(
      http.get('/api/users', () => {
        return HttpResponse.json([
          { id: '1', name: 'John' },
          { id: '2', name: 'Jane' },
        ])
      })
    )

    renderWithQuery(<UserList />)

    await waitFor(() => {
      expect(screen.getByText('John')).toBeInTheDocument()
      expect(screen.getByText('Jane')).toBeInTheDocument()
    })
  })
})
```
