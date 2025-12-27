# Debugging Guide

## Table of Contents
1. [DevTools Setup](#devtools-setup)
2. [Performance Profiling](#performance-profiling)
3. [Memory Leak Detection](#memory-leak-detection)
4. [Network Debugging](#network-debugging)
5. [State Debugging](#state-debugging)
6. [Common Error Patterns](#common-error-patterns)

## DevTools Setup

### Essential DevTools Extensions
```bash
# Install via npm untuk CI/testing
pnpm add -D @tanstack/react-query-devtools @tanstack/router-devtools
```

```typescript
// app/providers.tsx
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { TanStackRouterDevtools } from '@tanstack/router-devtools'

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {process.env.NODE_ENV === 'development' && (
        <>
          <ReactQueryDevtools initialIsOpen={false} />
          <TanStackRouterDevtools position="bottom-right" />
        </>
      )}
    </QueryClientProvider>
  )
}
```

### React DevTools Profiler Setup
```typescript
// Wrap specific components untuk profiling
import { Profiler } from 'react'

function onRenderCallback(
  id: string,
  phase: 'mount' | 'update',
  actualDuration: number,
  baseDuration: number,
  startTime: number,
  commitTime: number
) {
  if (actualDuration > 16) { // > 1 frame (60fps)
    console.warn(`Slow render: ${id} took ${actualDuration}ms`)
  }
}

function App() {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <Dashboard />
    </Profiler>
  )
}
```

## Performance Profiling

### Identify Unnecessary Re-renders

```typescript
// Debug hook untuk track re-renders
function useWhyDidYouUpdate(name: string, props: Record<string, any>) {
  const previousProps = useRef<Record<string, any>>()

  useEffect(() => {
    if (previousProps.current) {
      const allKeys = Object.keys({ ...previousProps.current, ...props })
      const changedProps: Record<string, any> = {}

      allKeys.forEach((key) => {
        if (previousProps.current![key] !== props[key]) {
          changedProps[key] = {
            from: previousProps.current![key],
            to: props[key],
          }
        }
      })

      if (Object.keys(changedProps).length) {
        console.log('[why-did-you-update]', name, changedProps)
      }
    }

    previousProps.current = props
  })
}

// Usage
function UserCard(props: UserCardProps) {
  useWhyDidYouUpdate('UserCard', props)
  // ...
}
```

### React DevTools Highlight Updates
1. Buka React DevTools
2. Settings (gear icon) → "Highlight updates when components render"
3. Interaksi dengan app → warna highlight menunjukkan re-render

### Profiler Recording
```typescript
// Record performance profile
import { startTransition } from 'react'

function handleHeavyOperation() {
  // Wrap non-urgent updates
  startTransition(() => {
    setLargeList(computeExpensiveList())
  })
}
```

## Memory Leak Detection

### Common Leak Patterns

```typescript
// ❌ MEMORY LEAK: Unsubscribed event listener
useEffect(() => {
  window.addEventListener('resize', handleResize)
  // Missing cleanup!
}, [])

// ✅ FIXED: Proper cleanup
useEffect(() => {
  window.addEventListener('resize', handleResize)
  return () => window.removeEventListener('resize', handleResize)
}, [])

// ❌ MEMORY LEAK: Subscription tanpa unsubscribe
useEffect(() => {
  const subscription = eventEmitter.subscribe(handler)
  // Missing cleanup!
}, [])

// ✅ FIXED
useEffect(() => {
  const subscription = eventEmitter.subscribe(handler)
  return () => subscription.unsubscribe()
}, [])

// ❌ MEMORY LEAK: setInterval tanpa clear
useEffect(() => {
  setInterval(() => {
    fetchData()
  }, 5000)
}, [])

// ✅ FIXED
useEffect(() => {
  const intervalId = setInterval(() => {
    fetchData()
  }, 5000)
  return () => clearInterval(intervalId)
}, [])

// ❌ MEMORY LEAK: State update setelah unmount
useEffect(() => {
  fetchData().then(data => {
    setData(data) // Component mungkin sudah unmount!
  })
}, [])

// ✅ FIXED dengan AbortController
useEffect(() => {
  const controller = new AbortController()
  
  fetchData({ signal: controller.signal })
    .then(data => setData(data))
    .catch(err => {
      if (err.name !== 'AbortError') throw err
    })
  
  return () => controller.abort()
}, [])

// ✅ BETTER: Use TanStack Query (handles this automatically)
const { data } = useQuery({
  queryKey: ['data'],
  queryFn: fetchData,
})
```

### Heap Snapshot Analysis
1. Chrome DevTools → Memory tab
2. Take heap snapshot
3. Interaksi dengan app (navigate, open modals, etc)
4. Take another snapshot
5. Compare snapshots → cari objects yang tidak di-garbage collect

### Detect Leaks dengan Query Devtools
```typescript
// Check for abandoned queries
// Di Query Devtools, cari queries yang:
// - stale tapi tidak active
// - gcTime expired tapi masih ada

// Monitor query cache size
const queryClient = useQueryClient()
console.log('Cache size:', queryClient.getQueryCache().getAll().length)
```

## Network Debugging

### Request Interceptor untuk Debug

```typescript
// lib/api-client.ts
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
})

// Request logging
api.interceptors.request.use((config) => {
  if (process.env.NODE_ENV === 'development') {
    console.group(`🌐 ${config.method?.toUpperCase()} ${config.url}`)
    console.log('Params:', config.params)
    console.log('Data:', config.data)
    console.groupEnd()
  }
  return config
})

// Response logging
api.interceptors.response.use(
  (response) => {
    if (process.env.NODE_ENV === 'development') {
      console.group(`✅ ${response.config.url}`)
      console.log('Status:', response.status)
      console.log('Data:', response.data)
      console.groupEnd()
    }
    return response
  },
  (error) => {
    if (process.env.NODE_ENV === 'development') {
      console.group(`❌ ${error.config?.url}`)
      console.log('Status:', error.response?.status)
      console.log('Error:', error.response?.data)
      console.groupEnd()
    }
    return Promise.reject(error)
  }
)
```

### Mock Service Worker untuk Debug

```typescript
// mocks/handlers.ts
import { http, HttpResponse, delay } from 'msw'

export const handlers = [
  // Simulate slow network
  http.get('/api/users', async () => {
    await delay(2000) // 2 second delay
    return HttpResponse.json([])
  }),

  // Simulate error
  http.post('/api/users', () => {
    return HttpResponse.json(
      { message: 'Server error' },
      { status: 500 }
    )
  }),
]
```

## State Debugging

### Query State Inspector

```typescript
function QueryDebugger() {
  const queryClient = useQueryClient()
  const queries = queryClient.getQueryCache().getAll()

  return (
    <div className="fixed bottom-0 left-0 bg-black text-white p-4 text-xs max-h-64 overflow-auto">
      <h3>Query Cache ({queries.length} queries)</h3>
      {queries.map((query) => (
        <div key={query.queryHash}>
          <strong>{query.queryKey.join(' → ')}</strong>
          <span className={`ml-2 ${
            query.state.status === 'success' ? 'text-green-400' :
            query.state.status === 'error' ? 'text-red-400' :
            'text-yellow-400'
          }`}>
            {query.state.status}
          </span>
          <span className="ml-2 text-gray-400">
            {query.state.dataUpdatedAt
              ? `Updated ${new Date(query.state.dataUpdatedAt).toLocaleTimeString()}`
              : 'Never fetched'}
          </span>
        </div>
      ))}
    </div>
  )
}
```

### Zustand State Logger

```typescript
import { devtools } from 'zustand/middleware'

const useStore = create(
  devtools(
    (set) => ({
      count: 0,
      increment: () => set((state) => ({ count: state.count + 1 }), false, 'increment'),
    }),
    { name: 'MyStore' }
  )
)
```

## Common Error Patterns

### Error Boundaries

```typescript
// shared/components/ErrorBoundary.tsx
import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log ke error tracking service
    console.error('Error boundary caught:', error, errorInfo)
    this.props.onError?.(error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="error-fallback">
          <h2>Something went wrong</h2>
          <pre>{this.state.error?.message}</pre>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            Try again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

// Usage
<ErrorBoundary
  fallback={<ErrorPage />}
  onError={(error) => sendToErrorTracking(error)}
>
  <App />
</ErrorBoundary>
```

### Debug Hydration Mismatches

```typescript
// Suppress hydration warning (use sparingly!)
<time
  dateTime={date.toISOString()}
  suppressHydrationWarning
>
  {date.toLocaleDateString()} {/* Different on server vs client */}
</time>

// Better: Use consistent rendering
function FormattedDate({ date }: { date: Date }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return <time dateTime={date.toISOString()}>Loading...</time>
  }

  return (
    <time dateTime={date.toISOString()}>
      {date.toLocaleDateString()}
    </time>
  )
}
```

### Debug Production Errors

```typescript
// Error tracking integration
import * as Sentry from '@sentry/react'

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  integrations: [
    Sentry.browserTracingIntegration(),
    Sentry.replayIntegration(),
  ],
  tracesSampleRate: 0.1,
  replaysSessionSampleRate: 0.1,
})

// Wrap app
export function App() {
  return (
    <Sentry.ErrorBoundary fallback={<ErrorPage />}>
      <Router />
    </Sentry.ErrorBoundary>
  )
}
```
