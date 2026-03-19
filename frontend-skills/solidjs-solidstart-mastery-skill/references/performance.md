# Performance Optimization & Profiling

## Bundle Optimization

### Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import solid from 'vite-plugin-solid';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    solid(),
    visualizer({
      filename: 'dist/stats.html',
      open: true,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
  build: {
    target: 'esnext',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.debug'],
      },
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-solid': ['solid-js', 'solid-js/web', 'solid-js/store'],
          'vendor-tanstack': ['@tanstack/solid-query', '@tanstack/solid-table'],
          'vendor-utils': ['date-fns', 'zod', 'ky'],
        },
      },
    },
    chunkSizeWarningLimit: 500,
  },
});
```

### Dynamic Imports & Code Splitting

```typescript
import { lazy, Suspense } from 'solid-js';

// ✅ Route-level code splitting
const Dashboard = lazy(() => import('./routes/Dashboard'));
const Settings = lazy(() => import('./routes/Settings'));
const Analytics = lazy(() => import('./routes/Analytics'));

// ✅ Component-level lazy loading
const HeavyChart = lazy(() => import('./components/HeavyChart'));
const DataGrid = lazy(() => import('./components/DataGrid'));

// ✅ Feature-based splitting
const AdminPanel = lazy(() => 
  import('./features/admin').then(m => ({ default: m.AdminPanel }))
);

// ✅ Conditional loading
function ConditionalFeature() {
  const [showAdvanced, setShowAdvanced] = createSignal(false);
  
  return (
    <div>
      <button onClick={() => setShowAdvanced(true)}>
        Load Advanced Features
      </button>
      <Show when={showAdvanced()}>
        <Suspense fallback={<Spinner />}>
          <HeavyChart />
        </Suspense>
      </Show>
    </div>
  );
}
```

### Tree Shaking

```typescript
// ✅ Named exports for better tree shaking
// lib/utils/index.ts
export { formatDate } from './date';
export { formatCurrency } from './currency';
export { debounce, throttle } from './timing';

// ❌ Avoid barrel exports with side effects
// This bundles everything:
import * as utils from './utils';

// ✅ Import only what you need
import { formatDate } from './utils/date';
import { debounce } from './utils/timing';
```

### Icon Optimization

```typescript
// ✅ Use unplugin-icons for tree-shakeable icons
// vite.config.ts
import Icons from 'unplugin-icons/vite';

export default defineConfig({
  plugins: [
    Icons({
      compiler: 'solid',
      autoInstall: true,
    }),
  ],
});

// Usage - only bundles used icons
import IconHome from '~icons/lucide/home';
import IconUser from '~icons/lucide/user';
import IconSettings from '~icons/lucide/settings';

// ❌ Avoid: Imports entire icon library
import { Home, User, Settings } from 'lucide-solid';
```

## Render Performance

### Avoiding Unnecessary Re-renders

```typescript
// ❌ BAD: Inline objects create new references
function BadComponent() {
  const [name, setName] = createSignal('John');
  
  return <UserCard user={{ name: name(), age: 25 }} />; // New object every time!
}

// ✅ GOOD: Use signals directly or stores
function GoodComponent() {
  const [user, setUser] = createStore({ name: 'John', age: 25 });
  
  return <UserCard user={user} />; // Same reference, fine-grained updates
}

// ✅ GOOD: Pass signals as props
function GoodComponent2() {
  const [name, setName] = createSignal('John');
  const [age, setAge] = createSignal(25);
  
  return <UserCard name={name} age={age} />;
}
```

### Optimizing Lists

```typescript
// Use <Index> when items are static, values change
// Items maintain position, values are reactive
function StaticListOptimized() {
  const [items, setItems] = createSignal([
    { id: 1, value: 0 },
    { id: 2, value: 0 },
  ]);

  return (
    <Index each={items()}>
      {(item, index) => (
        <div>
          Item {index}: {item().value}
          <button onClick={() => 
            setItems(prev => prev.map((i, idx) => 
              idx === index ? { ...i, value: i.value + 1 } : i
            ))
          }>
            Increment
          </button>
        </div>
      )}
    </Index>
  );
}

// Use <For> when items can be added/removed/reordered
// Items are keyed, can be reorganized
function DynamicListOptimized() {
  const [todos, setTodos] = createSignal<Todo[]>([]);

  return (
    <For each={todos()}>
      {(todo) => (
        <TodoItem 
          todo={todo} 
          onDelete={() => setTodos(prev => prev.filter(t => t.id !== todo.id))}
        />
      )}
    </For>
  );
}
```

### Virtualization for Long Lists

```typescript
// Using @tanstack/solid-virtual
import { createVirtualizer } from '@tanstack/solid-virtual';

function VirtualizedList() {
  let parentRef: HTMLDivElement;
  const [items] = createSignal(Array.from({ length: 10000 }, (_, i) => ({
    id: i,
    name: `Item ${i}`,
  })));

  const virtualizer = createVirtualizer({
    get count() { return items().length; },
    getScrollElement: () => parentRef,
    estimateSize: () => 50,
    overscan: 5,
  });

  return (
    <div
      ref={parentRef!}
      style={{ height: '400px', overflow: 'auto' }}
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        <For each={virtualizer.getVirtualItems()}>
          {(virtualRow) => (
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              {items()[virtualRow.index].name}
            </div>
          )}
        </For>
      </div>
    </div>
  );
}
```

### Memoization

```typescript
import { createMemo } from 'solid-js';

function ExpensiveComponent(props: { data: DataPoint[] }) {
  // ✅ Memoize expensive computations
  const processedData = createMemo(() => {
    console.log('Processing data...'); // Only runs when data changes
    return props.data
      .filter(d => d.value > 0)
      .map(d => ({ ...d, normalized: d.value / maxValue }))
      .sort((a, b) => b.normalized - a.normalized);
  });

  // ✅ Memoize derived values
  const summary = createMemo(() => ({
    total: processedData().reduce((sum, d) => sum + d.value, 0),
    count: processedData().length,
    average: processedData().length > 0 
      ? processedData().reduce((sum, d) => sum + d.value, 0) / processedData().length 
      : 0,
  }));

  return (
    <div>
      <p>Total: {summary().total}</p>
      <For each={processedData()}>
        {(item) => <DataRow item={item} />}
      </For>
    </div>
  );
}
```

## Image Optimization

```typescript
// ✅ Lazy loading images
function LazyImage(props: { src: string; alt: string }) {
  return (
    <img
      src={props.src}
      alt={props.alt}
      loading="lazy"
      decoding="async"
    />
  );
}

// ✅ Responsive images with srcset
function ResponsiveImage(props: { src: string; alt: string }) {
  const getSrcSet = () => {
    const base = props.src.replace(/\.[^.]+$/, '');
    const ext = props.src.split('.').pop();
    return `
      ${base}-320.${ext} 320w,
      ${base}-640.${ext} 640w,
      ${base}-1280.${ext} 1280w
    `;
  };

  return (
    <img
      src={props.src}
      srcset={getSrcSet()}
      sizes="(max-width: 320px) 280px, (max-width: 640px) 600px, 1200px"
      alt={props.alt}
      loading="lazy"
    />
  );
}

// ✅ Using picture element for modern formats
function ModernImage(props: { src: string; alt: string }) {
  const base = props.src.replace(/\.[^.]+$/, '');
  
  return (
    <picture>
      <source srcset={`${base}.avif`} type="image/avif" />
      <source srcset={`${base}.webp`} type="image/webp" />
      <img src={props.src} alt={props.alt} loading="lazy" />
    </picture>
  );
}
```

## Network Optimization

### Prefetching

```typescript
// Route prefetching with SolidStart
import { A } from '@solidjs/router';

// ✅ Prefetch on hover (default)
<A href="/dashboard">Dashboard</A>

// ✅ Prefetch on mount
<A href="/dashboard" preload="load">Dashboard</A>

// ✅ Custom prefetch
function PrefetchLink(props: { href: string; children: JSX.Element }) {
  let linkRef: HTMLAnchorElement;
  
  const prefetch = () => {
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = props.href;
    document.head.appendChild(link);
  };
  
  onMount(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          prefetch();
          observer.disconnect();
        }
      },
      { rootMargin: '100px' }
    );
    observer.observe(linkRef);
    onCleanup(() => observer.disconnect());
  });
  
  return <a ref={linkRef!} href={props.href}>{props.children}</a>;
}
```

### Request Deduplication with TanStack Query

```typescript
// TanStack Query automatically dedupes requests
const query = createQuery(() => ({
  queryKey: ['users', userId()],
  queryFn: () => fetchUser(userId()),
  staleTime: 5 * 60 * 1000, // Don't refetch for 5 minutes
}));

// Multiple components using same query = 1 request
function UserName() {
  const query = useUser(userId); // Uses cache
  return <span>{query.data?.name}</span>;
}

function UserAvatar() {
  const query = useUser(userId); // Uses same cache
  return <img src={query.data?.avatar} />;
}
```

## Profiling Tools

### Browser DevTools

```typescript
// Performance marks and measures
function ProfiledComponent() {
  onMount(() => {
    performance.mark('component-start');
  });

  createEffect(() => {
    performance.mark('effect-start');
    // ... effect logic
    performance.mark('effect-end');
    performance.measure('effect-duration', 'effect-start', 'effect-end');
  });

  return (
    <div
      ref={(el) => {
        performance.mark('component-end');
        performance.measure('component-render', 'component-start', 'component-end');
        
        const measures = performance.getEntriesByType('measure');
        console.table(measures);
      }}
    >
      Content
    </div>
  );
}
```

### Custom Performance Hook

```typescript
// hooks/usePerformance.ts
export function usePerformance(name: string) {
  const metrics = {
    renderTime: 0,
    effectTime: 0,
    updateCount: 0,
  };

  let renderStart: number;

  onMount(() => {
    renderStart = performance.now();
    
    requestAnimationFrame(() => {
      metrics.renderTime = performance.now() - renderStart;
      console.log(`[${name}] Initial render: ${metrics.renderTime.toFixed(2)}ms`);
    });
  });

  const trackEffect = (effectName: string, fn: () => void) => {
    createEffect(() => {
      const start = performance.now();
      fn();
      const duration = performance.now() - start;
      metrics.updateCount++;
      
      if (duration > 16) { // Longer than 1 frame
        console.warn(`[${name}] Slow effect "${effectName}": ${duration.toFixed(2)}ms`);
      }
    });
  };

  return { metrics, trackEffect };
}
```

### Bundle Analysis Script

```json
// package.json
{
  "scripts": {
    "build": "vinxi build",
    "build:analyze": "vinxi build && npx source-map-explorer dist/**/*.js",
    "build:stats": "vinxi build --mode production",
    "lighthouse": "lighthouse http://localhost:3000 --output html --output-path ./lighthouse-report.html"
  }
}
```

## Critical Rendering Path

### Inline Critical CSS

```typescript
// app.config.ts
import { defineConfig } from '@solidjs/start/config';

export default defineConfig({
  server: {
    // Inline critical CSS
    inlineDynamicImports: true,
  },
  vite: {
    build: {
      cssCodeSplit: true,
      cssMinify: 'lightningcss',
    },
  },
});
```

### Preload Critical Resources

```typescript
// entry-server.tsx
export default function handleRequest(
  request: Request,
  { manifest }: { manifest: Manifest }
) {
  return renderToStream(() => <App />, {
    onCompleteShell() {
      // Preload critical resources
      return `
        <link rel="preload" href="/fonts/inter.woff2" as="font" type="font/woff2" crossorigin>
        <link rel="preload" href="/api/user" as="fetch" crossorigin>
      `;
    },
  });
}
```

## Memory Optimization

### Cleanup Patterns

```typescript
// ✅ Always cleanup subscriptions
function Component() {
  createEffect(() => {
    const ws = new WebSocket('/ws');
    const handler = (e: MessageEvent) => processMessage(e.data);
    
    ws.addEventListener('message', handler);
    
    onCleanup(() => {
      ws.removeEventListener('message', handler);
      ws.close();
    });
  });
}

// ✅ Cleanup intervals and timeouts
function TimerComponent() {
  const [count, setCount] = createSignal(0);
  
  createEffect(() => {
    const interval = setInterval(() => {
      setCount(c => c + 1);
    }, 1000);
    
    onCleanup(() => clearInterval(interval));
  });
  
  return <span>{count()}</span>;
}

// ✅ Abort fetch requests
function FetchComponent(props: { url: string }) {
  const [data, setData] = createSignal(null);
  
  createEffect(() => {
    const controller = new AbortController();
    
    fetch(props.url, { signal: controller.signal })
      .then(r => r.json())
      .then(setData)
      .catch(e => {
        if (e.name !== 'AbortError') throw e;
      });
    
    onCleanup(() => controller.abort());
  });
  
  return <div>{JSON.stringify(data())}</div>;
}
```

### WeakMap for Caching

```typescript
// Use WeakMap for caching to allow garbage collection
const computationCache = new WeakMap<object, ComputedResult>();

function getCachedComputation(obj: object): ComputedResult {
  if (!computationCache.has(obj)) {
    computationCache.set(obj, expensiveComputation(obj));
  }
  return computationCache.get(obj)!;
}
```

## Performance Checklist

```markdown
## Pre-deployment Checklist

### Bundle
- [ ] Bundle size < 200KB gzipped (initial)
- [ ] Code splitting implemented for routes
- [ ] Tree shaking verified (no dead code)
- [ ] Source maps disabled in production

### Rendering
- [ ] No inline object creation in JSX
- [ ] Lists use <For> or <Index> appropriately
- [ ] Large lists virtualized
- [ ] Images lazy loaded

### Network
- [ ] API responses cached (TanStack Query)
- [ ] Critical resources preloaded
- [ ] Fonts optimized (subset, woff2)
- [ ] Images in modern formats (WebP/AVIF)

### Metrics
- [ ] LCP < 2.5s
- [ ] FID < 100ms
- [ ] CLS < 0.1
- [ ] TTI < 3.8s
```
