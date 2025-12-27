# Debugging Techniques & DevTools

## SolidJS DevTools

### Installation

```bash
# Install the DevTools
pnpm add -D solid-devtools

# For Vite projects
pnpm add -D vite-plugin-solid-devtools
```

### Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import solid from 'vite-plugin-solid';
import devtools from 'solid-devtools/vite';

export default defineConfig({
  plugins: [
    devtools({
      autoname: true,        // Automatically name signals
      locator: {
        targetIDE: 'vscode', // Click to open in VS Code
        key: 'Ctrl',         // Hold Ctrl to see component
        jsxLocation: true,
      },
    }),
    solid(),
  ],
});
```

### DevTools Features

```typescript
// 1. Named Signals (easier debugging)
const [count, setCount] = createSignal(0, { name: 'count' });
const [user, setUser] = createSignal(null, { name: 'currentUser' });

// 2. Named Effects
createEffect(() => {
  console.log('Count:', count());
}, { name: 'logCount' });

// 3. Debug utility
import { createEffect, DEV } from 'solid-js';

if (DEV) {
  createEffect(() => {
    console.log('[DEV] State changed:', someSignal());
  });
}
```

## Reactivity Debugging

### Track Signal Dependencies

```typescript
// Debug which signals trigger effects
import { createEffect, createSignal, untrack } from 'solid-js';

function debugEffect(name: string, fn: () => void) {
  createEffect(() => {
    console.group(`[Effect] ${name}`);
    console.log('Running at:', new Date().toISOString());
    fn();
    console.groupEnd();
  });
}

// Usage
debugEffect('userProfile', () => {
  console.log('user:', user());
  console.log('preferences:', preferences());
});
```

### Detect Reactivity Loss

```typescript
// Common reactivity loss patterns

// ❌ Pattern 1: Destructuring props
function BadComponent(props: { name: string }) {
  const { name } = props; // ❌ Lost reactivity!
  return <span>{name}</span>; // Never updates
}

// ✅ Fix: Access directly
function GoodComponent(props: { name: string }) {
  return <span>{props.name}</span>; // ✅ Updates correctly
}

// ❌ Pattern 2: Early signal access
function BadComponent2() {
  const data = fetchData(); // ❌ Captured once!
  return <div>{data}</div>;
}

// ✅ Fix: Access in JSX
function GoodComponent2() {
  return <div>{fetchData()}</div>; // ✅ Reactive
}

// ❌ Pattern 3: Storing signal value
function BadComponent3() {
  const value = count(); // ❌ Snapshot, not reactive
  
  return <button onClick={() => console.log(value)}>
    Log: {value}
  </button>;
}

// ✅ Fix: Call signal in usage
function GoodComponent3() {
  return <button onClick={() => console.log(count())}>
    Log: {count()}
  </button>;
}
```

### Signal Dependency Graph

```typescript
// Visualize signal dependencies
import { createSignal, createMemo, createRoot, getOwner } from 'solid-js';

function logDependencyTree() {
  const owner = getOwner();
  if (!owner) return;
  
  console.log('Owner:', owner);
  console.log('Sources:', (owner as any).sources);
  console.log('Observers:', (owner as any).observers);
}

// Use in component
function DebugComponent() {
  const [a, setA] = createSignal(1);
  const [b, setB] = createSignal(2);
  const sum = createMemo(() => a() + b());
  
  onMount(() => {
    logDependencyTree();
  });
  
  return <div>{sum()}</div>;
}
```

## Performance Debugging

### Render Counting

```typescript
// Hook to count renders
function useRenderCount(componentName: string) {
  let count = 0;
  
  onMount(() => {
    count = 1;
    console.log(`[${componentName}] Initial render`);
  });
  
  // This effect runs on each reactive update
  createEffect(() => {
    // Access reactive dependencies you want to track
    count++;
    console.log(`[${componentName}] Render #${count}`);
  });
  
  return () => count;
}

// Usage
function MyComponent(props: Props) {
  useRenderCount('MyComponent');
  // Component code...
}
```

### Effect Profiling

```typescript
// Profile effect execution time
function profileEffect(name: string, fn: () => void) {
  createEffect(() => {
    const start = performance.now();
    fn();
    const end = performance.now();
    
    if (end - start > 16) { // More than 1 frame
      console.warn(`[SLOW EFFECT] ${name}: ${(end - start).toFixed(2)}ms`);
    }
  });
}

// Usage
profileEffect('expensiveComputation', () => {
  // Some expensive operation
  processLargeDataset(data());
});
```

### Memory Leak Detection

```typescript
// Check for cleanup issues
function createLeakDetector(name: string) {
  const instances = new Set<string>();
  
  return {
    track: (id: string) => {
      instances.add(id);
      console.log(`[${name}] Created: ${id}, Total: ${instances.size}`);
      
      onCleanup(() => {
        instances.delete(id);
        console.log(`[${name}] Cleaned: ${id}, Total: ${instances.size}`);
      });
    },
    getCount: () => instances.size,
    getInstances: () => [...instances],
  };
}

// Usage
const subscriptionTracker = createLeakDetector('WebSocket');

function ChatComponent(props: { roomId: string }) {
  const instanceId = `room-${props.roomId}-${Date.now()}`;
  subscriptionTracker.track(instanceId);
  
  const ws = new WebSocket(`/chat/${props.roomId}`);
  
  onCleanup(() => {
    ws.close();
  });
}
```

## Network Debugging

### API Request Logging

```typescript
// lib/api/debug-client.ts
import ky from 'ky';

const isDev = import.meta.env.DEV;

export const api = ky.create({
  prefixUrl: '/api',
  hooks: {
    beforeRequest: isDev ? [
      (request) => {
        console.group(`🌐 ${request.method} ${request.url}`);
        console.log('Headers:', Object.fromEntries(request.headers));
        console.log('Time:', new Date().toISOString());
      }
    ] : [],
    afterResponse: isDev ? [
      async (request, options, response) => {
        const duration = performance.now();
        const body = await response.clone().json().catch(() => null);
        
        console.log('Status:', response.status);
        console.log('Response:', body);
        console.log('Duration:', `${duration.toFixed(2)}ms`);
        console.groupEnd();
      }
    ] : [],
    beforeError: [
      (error) => {
        console.error('❌ Request failed:', error.message);
        console.error('URL:', error.request?.url);
        console.error('Response:', error.response);
        return error;
      }
    ],
  },
});
```

### TanStack Query Debugging

```typescript
// Enable query devtools
import { QueryClientProvider } from '@tanstack/solid-query';
import { SolidQueryDevtools } from '@tanstack/solid-query-devtools';

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* Your app */}
      <SolidQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

// Log query state changes
import { createQuery } from '@tanstack/solid-query';

const query = createQuery(() => ({
  queryKey: ['users'],
  queryFn: fetchUsers,
  onError: (error) => console.error('[Query Error]', error),
  onSuccess: (data) => console.log('[Query Success]', data),
  onSettled: (data, error) => console.log('[Query Settled]', { data, error }),
}));
```

## SSR Debugging

### Hydration Mismatch Detection

```typescript
// Detect hydration mismatches
import { isServer, hydrate } from 'solid-js/web';

function HydrationDebugger(props: ParentProps) {
  if (!isServer) {
    const serverHTML = document.getElementById('app')?.innerHTML;
    
    onMount(() => {
      const clientHTML = document.getElementById('app')?.innerHTML;
      
      if (serverHTML !== clientHTML) {
        console.warn('⚠️ Hydration mismatch detected!');
        console.log('Server HTML:', serverHTML?.slice(0, 500));
        console.log('Client HTML:', clientHTML?.slice(0, 500));
      }
    });
  }
  
  return props.children;
}

// Common causes:
// 1. Using Date.now() or Math.random() during render
// 2. Accessing window/document during SSR
// 3. Different timezone between server/client
// 4. Browser extensions modifying DOM
```

### Server-Side Logging

```typescript
// routes/api/debug.ts
import type { APIEvent } from '@solidjs/start/server';

export async function GET(event: APIEvent) {
  const headers = Object.fromEntries(event.request.headers);
  const url = new URL(event.request.url);
  
  console.log('=== Server Debug ===');
  console.log('URL:', url.pathname);
  console.log('Query:', Object.fromEntries(url.searchParams));
  console.log('Headers:', headers);
  console.log('Cookies:', event.request.headers.get('cookie'));
  
  return new Response(JSON.stringify({
    timestamp: new Date().toISOString(),
    path: url.pathname,
    query: Object.fromEntries(url.searchParams),
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
}
```

## Error Tracking

### Global Error Handler

```typescript
// lib/error-handler.ts
import { ErrorBoundary, onError } from 'solid-js';

interface ErrorReport {
  message: string;
  stack?: string;
  componentStack?: string;
  timestamp: string;
  url: string;
  userAgent: string;
}

function reportError(error: ErrorReport) {
  console.error('Error Report:', error);
  
  // Send to error tracking service
  if (import.meta.env.PROD) {
    fetch('/api/errors', {
      method: 'POST',
      body: JSON.stringify(error),
    }).catch(console.error);
  }
}

// Global error boundary
export function AppErrorBoundary(props: ParentProps) {
  return (
    <ErrorBoundary
      fallback={(err, reset) => {
        reportError({
          message: err.message,
          stack: err.stack,
          timestamp: new Date().toISOString(),
          url: window.location.href,
          userAgent: navigator.userAgent,
        });
        
        return (
          <div class="error-fallback">
            <h1>Something went wrong</h1>
            <pre>{err.message}</pre>
            <button onClick={reset}>Try Again</button>
          </div>
        );
      }}
    >
      {props.children}
    </ErrorBoundary>
  );
}

// Catch unhandled errors
if (typeof window !== 'undefined') {
  window.onerror = (message, source, line, col, error) => {
    reportError({
      message: String(message),
      stack: error?.stack,
      timestamp: new Date().toISOString(),
      url: `${source}:${line}:${col}`,
      userAgent: navigator.userAgent,
    });
  };
  
  window.onunhandledrejection = (event) => {
    reportError({
      message: event.reason?.message || 'Unhandled Promise Rejection',
      stack: event.reason?.stack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent,
    });
  };
}
```

## Console Utilities

```typescript
// lib/debug.ts
export const debug = {
  signal: (name: string, value: unknown) => {
    console.log(`%c[Signal] ${name}:`, 'color: #61dafb; font-weight: bold', value);
  },
  
  effect: (name: string) => {
    console.log(`%c[Effect] ${name} triggered`, 'color: #ffd700; font-weight: bold');
  },
  
  render: (component: string) => {
    console.log(`%c[Render] ${component}`, 'color: #98c379; font-weight: bold');
  },
  
  api: (method: string, url: string, data?: unknown) => {
    console.log(`%c[API] ${method} ${url}`, 'color: #c678dd; font-weight: bold', data);
  },
  
  warn: (message: string, data?: unknown) => {
    console.warn(`%c[Warning] ${message}`, 'color: #e5c07b; font-weight: bold', data);
  },
  
  error: (message: string, error?: Error) => {
    console.error(`%c[Error] ${message}`, 'color: #e06c75; font-weight: bold', error);
  },
  
  table: (data: unknown[], columns?: string[]) => {
    console.table(data, columns);
  },
  
  time: (label: string) => console.time(label),
  timeEnd: (label: string) => console.timeEnd(label),
  
  group: (label: string) => console.group(label),
  groupEnd: () => console.groupEnd(),
};

// Usage
debug.signal('user', user());
debug.effect('fetchUserData');
debug.api('GET', '/api/users', { limit: 10 });
```

## VS Code Debug Configuration

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "chrome",
      "request": "launch",
      "name": "Debug SolidStart",
      "url": "http://localhost:3000",
      "webRoot": "${workspaceFolder}",
      "sourceMaps": true,
      "sourceMapPathOverrides": {
        "webpack:///*": "${workspaceFolder}/*"
      }
    },
    {
      "type": "node",
      "request": "launch",
      "name": "Debug Server",
      "program": "${workspaceFolder}/node_modules/vinxi/bin/vinxi.mjs",
      "args": ["dev"],
      "cwd": "${workspaceFolder}",
      "env": {
        "DEBUG": "*"
      }
    }
  ]
}
```
