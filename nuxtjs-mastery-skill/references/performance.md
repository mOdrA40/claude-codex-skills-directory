# Performance Optimization — Vue.js & Nuxt 3

> Performance bukan afterthought. Build with performance in mind dari hari pertama.

## Table of Contents
1. [Performance Metrics](#performance-metrics)
2. [Bundle Optimization](#bundle-optimization)
3. [Runtime Optimization](#runtime-optimization)
4. [Network Optimization](#network-optimization)
5. [Rendering Optimization](#rendering-optimization)
6. [Profiling & Monitoring](#profiling--monitoring)

---

## Performance Metrics

### Target Metrics

| Metric | Target | Why |
|--------|--------|-----|
| LCP (Largest Contentful Paint) | < 2.5s | User-perceived load time |
| FID (First Input Delay) | < 100ms | Interactivity |
| CLS (Cumulative Layout Shift) | < 0.1 | Visual stability |
| TTI (Time to Interactive) | < 3.8s | Usable |
| Initial JS Bundle | < 200KB | Fast parse/execute |
| Total Page Weight | < 1MB | Mobile-friendly |

### Measuring Performance

```typescript
// composables/usePerformance.ts
export function usePerformance() {
  const metrics = ref<PerformanceMetrics | null>(null)

  onMounted(() => {
    if (!('performance' in window)) return

    // Core Web Vitals
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        console.log(`${entry.name}: ${entry.startTime}ms`)
      }
    })

    observer.observe({ type: 'largest-contentful-paint', buffered: true })
    observer.observe({ type: 'first-input', buffered: true })
    observer.observe({ type: 'layout-shift', buffered: true })

    // Navigation timing
    const timing = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    if (timing) {
      metrics.value = {
        dns: timing.domainLookupEnd - timing.domainLookupStart,
        connection: timing.connectEnd - timing.connectStart,
        ttfb: timing.responseStart - timing.requestStart,
        download: timing.responseEnd - timing.responseStart,
        domParsing: timing.domInteractive - timing.responseEnd,
        domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
        loaded: timing.loadEventEnd - timing.navigationStart,
      }
    }
  })

  return { metrics }
}
```

---

## Bundle Optimization

### Tree Shaking

```typescript
// ❌ BAD: Import entire library
import _ from 'lodash' // 70KB!
import * as dateFns from 'date-fns' // 80KB!

// ✅ GOOD: Import specific functions
import debounce from 'lodash-es/debounce' // 1KB
import { format, parseISO } from 'date-fns' // 5KB
```

### Dynamic Imports

```typescript
// ❌ BAD: Import semua di awal
import HeavyChart from '~/components/HeavyChart.vue'
import PdfViewer from '~/components/PdfViewer.vue'

// ✅ GOOD: Lazy load heavy components
const HeavyChart = defineAsyncComponent(() =>
  import('~/components/HeavyChart.vue')
)

// Dengan loading state
const PdfViewer = defineAsyncComponent({
  loader: () => import('~/components/PdfViewer.vue'),
  loadingComponent: LoadingSpinner,
  delay: 200,
  errorComponent: ErrorDisplay,
  timeout: 30000,
})
```

### Code Splitting Routes

```typescript
// Nuxt auto-splits by default
// pages/dashboard/analytics.vue akan jadi chunk terpisah

// Untuk manual control:
// nuxt.config.ts
export default defineNuxtConfig({
  experimental: {
    payloadExtraction: true,
  },
  // Split vendor chunks
  vite: {
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            'vue-vendor': ['vue', 'vue-router', 'pinia'],
            'query-vendor': ['@tanstack/vue-query'],
          },
        },
      },
    },
  },
})
```

### Analyze Bundle

```bash
# Install analyzer
npm install -D nuxt-bundle-analyze

# nuxt.config.ts
export default defineNuxtConfig({
  modules: ['nuxt-bundle-analyze'],
})

# Run analysis
npx nuxt analyze
```

### External Dependencies

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  vite: {
    build: {
      rollupOptions: {
        external: [
          // Large libraries yang bisa dari CDN
        ],
      },
    },
  },
})

// Atau gunakan CDN untuk libraries besar
// app.vue
<script>
useHead({
  script: [
    {
      src: 'https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js',
      defer: true,
    },
  ],
})
</script>
```

---

## Runtime Optimization

### Computed vs Methods

```typescript
// ❌ BAD: Method di template = recalculate setiap render
<template>
  <div v-for="item in items">
    {{ formatPrice(item.price) }} <!-- Called N times per render! -->
  </div>
</template>

// ✅ GOOD: Precompute dengan computed
const formattedItems = computed(() =>
  items.value.map(item => ({
    ...item,
    formattedPrice: formatPrice(item.price),
  }))
)

<template>
  <div v-for="item in formattedItems">
    {{ item.formattedPrice }} <!-- Cached! -->
  </div>
</template>
```

### v-once untuk Static Content

```vue
<template>
  <!-- Content yang tidak pernah berubah -->
  <header v-once>
    <h1>My App</h1>
    <p>Welcome to the application</p>
  </header>
  
  <!-- Dynamic content -->
  <main>
    <ProductList :products="products" />
  </main>
</template>
```

### v-memo untuk Expensive Renders

```vue
<template>
  <!-- Re-render HANYA jika item.id atau selected berubah -->
  <div
    v-for="item in list"
    :key="item.id"
    v-memo="[item.id, item.id === selected]"
  >
    <ExpensiveComponent :item="item" :selected="item.id === selected" />
  </div>
</template>
```

### shallowRef untuk Large Objects

```typescript
// ❌ BAD: Deep reactivity untuk data besar
const hugeData = ref(arrayOf10000Items) // Proxy overhead!

// ✅ GOOD: Shallow reactivity
const hugeData = shallowRef(arrayOf10000Items)

// Update dengan replace (trigger reactivity)
hugeData.value = [...hugeData.value, newItem]

// Untuk objects yang tidak perlu reactive sama sekali
const staticConfig = markRaw(largeConfigObject)
```

### Virtual Scrolling

```typescript
// Untuk list dengan 1000+ items
import { useVirtualList } from '@vueuse/core'

const items = ref(Array.from({ length: 10000 }, (_, i) => ({
  id: i,
  name: `Item ${i}`,
})))

const { list, containerProps, wrapperProps } = useVirtualList(items, {
  itemHeight: 50,
})

<template>
  <div v-bind="containerProps" class="h-[400px] overflow-auto">
    <div v-bind="wrapperProps">
      <div v-for="{ data, index } in list" :key="index" class="h-[50px]">
        {{ data.name }}
      </div>
    </div>
  </div>
</template>
```

---

## Network Optimization

### Caching Strategy dengan TanStack Query

```typescript
// plugins/vue-query.ts
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Data jarang berubah: cache lama
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 30,   // 30 minutes
      
      // Disable unnecessary refetches
      refetchOnWindowFocus: false,
      refetchOnReconnect: 'always',
    },
  },
})

// Per-query customization
const { data } = useQuery({
  queryKey: ['user-preferences'],
  queryFn: fetchPreferences,
  staleTime: 1000 * 60 * 60, // 1 hour - jarang berubah
})

const { data: stockPrice } = useQuery({
  queryKey: ['stock', symbol],
  queryFn: () => fetchStock(symbol),
  staleTime: 0, // Selalu stale
  refetchInterval: 5000, // Poll setiap 5 detik
})
```

### Request Batching

```typescript
// Batch multiple requests
import { useQueries } from '@tanstack/vue-query'

const productQueries = useQueries({
  queries: productIds.value.map(id => ({
    queryKey: ['product', id],
    queryFn: () => fetchProduct(id),
    staleTime: 1000 * 60 * 5,
  })),
})

// Atau gunakan batch endpoint di server
const { data } = useQuery({
  queryKey: ['products', productIds],
  queryFn: () => $fetch('/api/products/batch', {
    method: 'POST',
    body: { ids: productIds.value },
  }),
})
```

### Prefetching

```typescript
// Prefetch on hover
const queryClient = useQueryClient()

function prefetchProduct(productId: string) {
  queryClient.prefetchQuery({
    queryKey: ['product', productId],
    queryFn: () => fetchProduct(productId),
    staleTime: 1000 * 60 * 5,
  })
}

<template>
  <ProductCard
    v-for="product in products"
    :key="product.id"
    :product="product"
    @mouseenter="prefetchProduct(product.id)"
  />
</template>

// Prefetch next page
const { data, fetchNextPage, hasNextPage } = useInfiniteProducts()

watch(data, () => {
  if (hasNextPage.value) {
    // Prefetch next page saat user scroll mendekati bawah
    fetchNextPage()
  }
})
```

### Image Optimization

```vue
<!-- Nuxt Image module -->
<template>
  <!-- Lazy loading dengan blur placeholder -->
  <NuxtImg
    src="/images/hero.jpg"
    width="800"
    height="600"
    loading="lazy"
    placeholder
    format="webp"
    quality="80"
    sizes="sm:100vw md:50vw lg:400px"
  />
  
  <!-- Responsive images -->
  <NuxtPicture
    src="/images/product.jpg"
    sizes="xs:100vw sm:50vw md:400px"
    format="webp,avif"
  />
</template>
```

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  modules: ['@nuxt/image'],
  image: {
    quality: 80,
    format: ['webp', 'avif'],
    screens: {
      xs: 320,
      sm: 640,
      md: 768,
      lg: 1024,
      xl: 1280,
    },
  },
})
```

---

## Rendering Optimization

### Keep-Alive untuk Route Caching

```vue
<!-- app.vue -->
<template>
  <NuxtLayout>
    <NuxtPage :keepalive="true" />
  </NuxtLayout>
</template>

<!-- Atau selective keep-alive -->
<template>
  <NuxtLayout>
    <NuxtPage
      :keepalive="{
        include: ['Dashboard', 'ProductList'],
        max: 5,
      }"
    />
  </NuxtLayout>
</template>

<!-- Di page level -->
<script setup>
definePageMeta({
  keepalive: true,
})
</script>
```

### Deferred Hydration

```vue
<!-- Hydrate saat visible -->
<template>
  <LazyHeavyComponent :hydrate-on-visible="true" />
</template>

<!-- Hydrate saat idle -->
<template>
  <LazyHeavyComponent :hydrate-on-idle="true" />
</template>

<!-- Hydrate on interaction -->
<template>
  <LazyHeavyComponent :hydrate-on-interaction="['click', 'focus']" />
</template>
```

### Suspense untuk Async Components

```vue
<template>
  <Suspense>
    <template #default>
      <AsyncDashboard />
    </template>
    
    <template #fallback>
      <DashboardSkeleton />
    </template>
  </Suspense>
</template>
```

### Prevent Layout Shifts

```css
/* Reserve space untuk dynamic content */
.image-container {
  aspect-ratio: 16 / 9;
  background-color: #f0f0f0;
}

/* Fixed dimensions untuk ads/embeds */
.ad-slot {
  width: 300px;
  height: 250px;
}

/* Skeleton dengan fixed height */
.skeleton-text {
  height: 1.2em;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}
```

---

## Profiling & Monitoring

### Vue DevTools Performance

```typescript
// Enable performance tracing
// app.vue
<script setup>
const app = useNuxtApp()

if (import.meta.dev) {
  app.vueApp.config.performance = true
}
</script>

// Di Vue DevTools > Performance tab:
// - Component render times
// - Re-render counts
// - Event timeline
```

### Custom Performance Marks

```typescript
// composables/usePerformanceMark.ts
export function usePerformanceMark(name: string) {
  const startMark = `${name}-start`
  const endMark = `${name}-end`

  function start() {
    performance.mark(startMark)
  }

  function end() {
    performance.mark(endMark)
    performance.measure(name, startMark, endMark)
    
    const measure = performance.getEntriesByName(name)[0]
    console.log(`[Perf] ${name}: ${measure.duration.toFixed(2)}ms`)
  }

  return { start, end }
}

// Usage
const { start, end } = usePerformanceMark('fetchProducts')

start()
await fetchProducts()
end()
// [Perf] fetchProducts: 234.56ms
```

### Web Vitals Reporting

```typescript
// plugins/web-vitals.client.ts
import { onCLS, onFID, onLCP, onTTFB, onINP } from 'web-vitals'

export default defineNuxtPlugin(() => {
  function sendToAnalytics(metric: any) {
    const body = JSON.stringify({
      name: metric.name,
      value: metric.value,
      rating: metric.rating, // 'good', 'needs-improvement', 'poor'
      delta: metric.delta,
      id: metric.id,
    })

    // Send to analytics
    if (navigator.sendBeacon) {
      navigator.sendBeacon('/api/analytics/vitals', body)
    }
  }

  onCLS(sendToAnalytics)
  onFID(sendToAnalytics)
  onLCP(sendToAnalytics)
  onTTFB(sendToAnalytics)
  onINP(sendToAnalytics)
})
```

### Performance Budget

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  vite: {
    build: {
      // Warn jika chunk > 500KB
      chunkSizeWarningLimit: 500,
    },
  },
  
  // Custom plugin untuk check bundle size
  hooks: {
    'build:done': async () => {
      const { statSync } = await import('fs')
      const jsBundle = statSync('.output/public/_nuxt/entry.js')
      
      if (jsBundle.size > 200 * 1024) {
        console.warn(`⚠️ Entry bundle too large: ${(jsBundle.size / 1024).toFixed(2)}KB`)
      }
    },
  },
})
```

### Lighthouse CI

```yaml
# .github/workflows/lighthouse.yml
name: Lighthouse CI
on: [push]
jobs:
  lighthouse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci && npm run build
      - name: Run Lighthouse
        uses: treosh/lighthouse-ci-action@v11
        with:
          urls: |
            http://localhost:3000
            http://localhost:3000/products
          budgetPath: ./lighthouse-budget.json
          uploadArtifacts: true
```

```json
// lighthouse-budget.json
[
  {
    "path": "/*",
    "timings": [
      { "metric": "first-contentful-paint", "budget": 2000 },
      { "metric": "interactive", "budget": 3800 },
      { "metric": "largest-contentful-paint", "budget": 2500 }
    ],
    "resourceSizes": [
      { "resourceType": "script", "budget": 200 },
      { "resourceType": "total", "budget": 1000 }
    ]
  }
]
```
