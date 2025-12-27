# Debugging Mastery untuk Vue.js & Nuxt 3

> Debugging adalah seni. Senior developer debug dengan SISTEM, bukan tebak-tebakan.

## Table of Contents
1. [Debugging Tools Setup](#debugging-tools-setup)
2. [Common Error Patterns](#common-error-patterns)
3. [Reactivity Debugging](#reactivity-debugging)
4. [Network Debugging](#network-debugging)
5. [Performance Profiling](#performance-profiling)
6. [Production Debugging](#production-debugging)

---

## Debugging Tools Setup

### Vue DevTools (WAJIB)

```bash
# Install Vue DevTools browser extension
# Chrome: https://chrome.google.com/webstore/detail/vuejs-devtools
# Firefox: https://addons.mozilla.org/en-US/firefox/addon/vue-js-devtools/
```

**Fitur yang HARUS dikuasai:**
- Component Inspector - Lihat props, state, computed
- Timeline - Track events dan mutations
- Pinia - State changes tracking
- Routes - Navigation debugging
- Performance - Component render times

### Nuxt DevTools

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  devtools: { 
    enabled: true,
    timeline: {
      enabled: true,
    },
  },
})
```

**Fitur Nuxt DevTools:**
- Components tree dengan props
- Composables state
- Server routes inspection
- Payload viewer untuk SSR
- Network tab untuk API calls

### TanStack Query DevTools

```typescript
// plugins/vue-query.ts
import { VueQueryPlugin } from '@tanstack/vue-query'

export default defineNuxtPlugin((nuxtApp) => {
  nuxtApp.vueApp.use(VueQueryPlugin, {
    queryClientConfig: {
      defaultOptions: {
        queries: {
          staleTime: 1000 * 60 * 5,
        },
      },
    },
    // Enable devtools
    enableDevtoolsV6Plugin: true,
  })
})
```

### Console Utilities

```typescript
// utils/debug.ts
const isDev = process.env.NODE_ENV === 'development'

export const debug = {
  log: (...args: any[]) => isDev && console.log('[DEBUG]', ...args),
  warn: (...args: any[]) => isDev && console.warn('[WARN]', ...args),
  error: (...args: any[]) => console.error('[ERROR]', ...args),
  
  // Group related logs
  group: (label: string, fn: () => void) => {
    if (!isDev) return
    console.group(label)
    fn()
    console.groupEnd()
  },
  
  // Table for arrays/objects
  table: (data: any) => isDev && console.table(data),
  
  // Time operations
  time: (label: string) => isDev && console.time(label),
  timeEnd: (label: string) => isDev && console.timeEnd(label),
  
  // Trace call stack
  trace: () => isDev && console.trace(),
}

// Usage
debug.group('Product Fetch', () => {
  debug.log('Filters:', filters)
  debug.log('Response:', data)
  debug.table(data.products)
})
```

---

## Common Error Patterns

### Error: "Cannot read properties of undefined"

```typescript
// ❌ Error terjadi
const { data } = useProducts()
const firstProduct = data.value.products[0] // 💥 Error!

// 🔍 DIAGNOSA:
// - data.value mungkin null/undefined saat initial render
// - products mungkin tidak ada di response

// ✅ SOLUSI 1: Optional chaining
const firstProduct = data.value?.products?.[0]

// ✅ SOLUSI 2: Default value dengan computed
const products = computed(() => data.value?.products ?? [])
const firstProduct = computed(() => products.value[0])

// ✅ SOLUSI 3: Guard di template
<template>
  <div v-if="data?.products?.length">
    <ProductCard :product="data.products[0]" />
  </div>
  <EmptyState v-else />
</template>
```

### Error: "Maximum recursive updates exceeded"

```typescript
// ❌ Infinite loop
const count = ref(0)
watch(count, () => {
  count.value++ // 💥 Triggers watch again!
})

// ❌ Infinite loop in computed
const items = ref([1, 2, 3])
const sorted = computed(() => {
  items.value.sort() // 💥 Mutates original, triggers re-compute!
  return items.value
})

// ✅ SOLUSI: Jangan mutate di watch/computed
const sorted = computed(() => {
  return [...items.value].sort() // Clone first!
})

// ✅ Untuk watch, gunakan flag atau nextTick
const isUpdating = ref(false)
watch(data, async (newVal) => {
  if (isUpdating.value) return
  isUpdating.value = true
  await doSomething()
  isUpdating.value = false
})
```

### Error: "Hydration mismatch"

```typescript
// ❌ Server/client menghasilkan HTML berbeda
<template>
  <div>{{ new Date().toLocaleString() }}</div> <!-- 💥 Different on server/client -->
</template>

// ❌ Menggunakan browser-only API di SSR
<script setup>
const width = window.innerWidth // 💥 window undefined di server
</script>

// ✅ SOLUSI 1: Client-only wrapper
<template>
  <ClientOnly>
    <div>{{ new Date().toLocaleString() }}</div>
  </ClientOnly>
</template>

// ✅ SOLUSI 2: onMounted untuk browser APIs
<script setup>
const width = ref(0)
onMounted(() => {
  width.value = window.innerWidth
})
</script>

// ✅ SOLUSI 3: useNuxtApp untuk environment check
const nuxtApp = useNuxtApp()
if (import.meta.client) {
  // Browser-only code
}
```

### Error: "[Vue warn]: Property was accessed during render but is not defined"

```typescript
// ❌ Mengakses variable yang tidak ada
<template>
  <div>{{ undefinedVariable }}</div> <!-- 💥 Not defined -->
</template>

// ✅ DIAGNOSA dengan Vue DevTools:
// 1. Buka Vue DevTools
// 2. Pilih component yang error
// 3. Check "Setup" tab - pastikan variable terdefinisi

// ✅ SOLUSI: Pastikan semua variable didefinisikan
<script setup>
const definedVariable = ref('value')
</script>
```

---

## Reactivity Debugging

### Debug Ref Changes

```typescript
// Composable untuk debug reactivity
export function useDebugRef<T>(
  ref: Ref<T>,
  name: string
) {
  if (process.env.NODE_ENV !== 'development') return

  watch(
    ref,
    (newVal, oldVal) => {
      console.group(`🔄 ${name} changed`)
      console.log('Old:', oldVal)
      console.log('New:', newVal)
      console.trace('Change origin:')
      console.groupEnd()
    },
    { deep: true }
  )
}

// Usage
const user = ref<User | null>(null)
useDebugRef(user, 'user')
```

### Detect Reactivity Loss

```typescript
// ❌ Reactivity loss - SANGAT COMMON BUG!
const { data } = useQuery({ /* ... */ })

// Loss 1: Destructuring reactive object
const { products } = data.value // 💥 products tidak reactive!

// Loss 2: Assigning to new variable
let items = data.value.products // 💥 items tidak reactive!

// Loss 3: Spread operator
const newData = { ...data.value } // 💥 newData tidak reactive!

// ✅ SOLUSI: Gunakan computed atau toRef
const products = computed(() => data.value?.products ?? [])

// Atau toRef untuk single property
const products = toRef(data.value, 'products')

// Untuk debugging, check dengan isRef/isReactive
import { isRef, isReactive } from 'vue'
console.log('Is reactive?', isReactive(data))
console.log('Is ref?', isRef(products))
```

### Watch Not Triggering

```typescript
// ❌ Watch tidak trigger
const filters = ref({
  category: 'all',
  page: 1
})

watch(filters, (newFilters) => {
  console.log('Filters changed!') // Tidak trigger saat filters.value.page = 2
})

// 🔍 DIAGNOSA:
// Object mutation tidak trigger shallow watch

// ✅ SOLUSI 1: deep watch
watch(filters, callback, { deep: true })

// ✅ SOLUSI 2: Watch specific property
watch(
  () => filters.value.page,
  (newPage) => console.log('Page changed:', newPage)
)

// ✅ SOLUSI 3: Replace entire object
filters.value = { ...filters.value, page: 2 }
```

---

## Network Debugging

### API Request Debugging

```typescript
// composables/useApi.ts
export function useApi() {
  const config = useRuntimeConfig()
  
  const api = $fetch.create({
    baseURL: config.public.apiBase,
    
    onRequest({ request, options }) {
      debug.group('🌐 API Request')
      debug.log('URL:', request)
      debug.log('Method:', options.method || 'GET')
      debug.log('Body:', options.body)
      debug.log('Headers:', options.headers)
      debug.groupEnd()
    },
    
    onRequestError({ request, error }) {
      debug.error('Request failed:', request, error)
    },
    
    onResponse({ request, response }) {
      debug.group('✅ API Response')
      debug.log('URL:', request)
      debug.log('Status:', response.status)
      debug.log('Data:', response._data)
      debug.groupEnd()
    },
    
    onResponseError({ request, response }) {
      debug.group('❌ API Error')
      debug.error('URL:', request)
      debug.error('Status:', response.status)
      debug.error('Data:', response._data)
      debug.groupEnd()
    },
  })
  
  return { api }
}
```

### TanStack Query Debugging

```typescript
// Debug specific query
const { data, error, status, fetchStatus } = useQuery({
  queryKey: ['products'],
  queryFn: async () => {
    debug.time('fetchProducts')
    const result = await $fetch('/api/products')
    debug.timeEnd('fetchProducts')
    return result
  },
})

// Watch query state changes
watch(
  () => ({ status: status.value, fetchStatus: fetchStatus.value }),
  (state) => {
    debug.log('Query state:', state)
  }
)

// Debug all queries via query client
const queryClient = useQueryClient()

// Log semua active queries
debug.log('Active queries:', queryClient.getQueryCache().getAll())

// Force refetch untuk testing
queryClient.invalidateQueries({ queryKey: ['products'] })
```

---

## Performance Profiling

### Component Render Tracking

```typescript
// composables/useRenderTracker.ts
export function useRenderTracker(componentName: string) {
  if (process.env.NODE_ENV !== 'development') return

  const renderCount = ref(0)
  
  onMounted(() => {
    debug.log(`[${componentName}] Mounted`)
  })
  
  onUpdated(() => {
    renderCount.value++
    debug.warn(`[${componentName}] Re-rendered (${renderCount.value}x)`)
  })
  
  onUnmounted(() => {
    debug.log(`[${componentName}] Unmounted after ${renderCount.value} renders`)
  })
  
  return { renderCount }
}

// Usage di component
const { renderCount } = useRenderTracker('ProductList')
```

### Memory Leak Detection

```typescript
// ❌ Common memory leaks
export function useBadComposable() {
  // Leak 1: Event listener tidak di-remove
  window.addEventListener('resize', handleResize)
  
  // Leak 2: Interval tidak di-clear
  setInterval(pollData, 5000)
  
  // Leak 3: Subscription tidak di-unsubscribe
  someObservable.subscribe(callback)
}

// ✅ SOLUSI: Cleanup di onUnmounted atau gunakan VueUse
export function useGoodComposable() {
  const handleResize = () => { /* ... */ }
  
  // VueUse auto-cleanup
  useEventListener('resize', handleResize)
  
  // Atau manual cleanup
  onMounted(() => {
    window.addEventListener('resize', handleResize)
  })
  
  onUnmounted(() => {
    window.removeEventListener('resize', handleResize)
  })
  
  // Interval dengan cleanup
  const intervalId = ref<NodeJS.Timeout>()
  
  onMounted(() => {
    intervalId.value = setInterval(pollData, 5000)
  })
  
  onUnmounted(() => {
    if (intervalId.value) {
      clearInterval(intervalId.value)
    }
  })
}

// Debug: Check for memory leaks
// Chrome DevTools > Memory > Take heap snapshot
// Bandingkan snapshot sebelum dan sesudah navigasi
```

### Bundle Analysis

```bash
# Install analyzer
npm install -D nuxt-bundle-analyze

# nuxt.config.ts
export default defineNuxtConfig({
  modules: ['nuxt-bundle-analyze'],
  bundleAnalyze: {
    enabled: true,
  },
})

# Run analysis
npm run analyze
```

---

## Production Debugging

### Error Tracking Setup

```typescript
// plugins/error-tracking.ts
export default defineNuxtPlugin((nuxtApp) => {
  // Global error handler
  nuxtApp.vueApp.config.errorHandler = (error, instance, info) => {
    console.error('Vue Error:', {
      error,
      component: instance?.$options?.name,
      info,
    })
    
    // Send to error tracking service (Sentry, etc.)
    // Sentry.captureException(error, { extra: { info } })
  }
  
  // Unhandled promise rejections
  if (import.meta.client) {
    window.addEventListener('unhandledrejection', (event) => {
      console.error('Unhandled Promise:', event.reason)
      // Sentry.captureException(event.reason)
    })
  }
})
```

### Source Maps untuk Production

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  sourcemap: {
    server: true,
    client: true, // Enable untuk debugging production issues
  },
  
  // Atau hanya untuk staging
  sourcemap: process.env.NODE_ENV !== 'production' ? {
    server: true,
    client: true,
  } : false,
})
```

### Debug Logging yang Aman

```typescript
// utils/logger.ts
type LogLevel = 'debug' | 'info' | 'warn' | 'error'

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
}

const currentLevel = process.env.NODE_ENV === 'production' ? 'error' : 'debug'

export const logger = {
  debug: (...args: any[]) => {
    if (LOG_LEVELS.debug >= LOG_LEVELS[currentLevel]) {
      console.debug('[DEBUG]', ...args)
    }
  },
  info: (...args: any[]) => {
    if (LOG_LEVELS.info >= LOG_LEVELS[currentLevel]) {
      console.info('[INFO]', ...args)
    }
  },
  warn: (...args: any[]) => {
    if (LOG_LEVELS.warn >= LOG_LEVELS[currentLevel]) {
      console.warn('[WARN]', ...args)
    }
  },
  error: (...args: any[]) => {
    console.error('[ERROR]', ...args)
    // Always log errors, consider sending to tracking service
  },
}
```

### Debug Flags

```typescript
// Runtime debug flags via URL
// https://myapp.com?debug=true&verbose=true

export function useDebugFlags() {
  const route = useRoute()
  
  const isDebug = computed(() => route.query.debug === 'true')
  const isVerbose = computed(() => route.query.verbose === 'true')
  
  return {
    isDebug,
    isVerbose,
    log: (...args: any[]) => {
      if (isDebug.value) console.log(...args)
    },
    verbose: (...args: any[]) => {
      if (isVerbose.value) console.log(...args)
    },
  }
}
```
