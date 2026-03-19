# TanStack Query Mastery untuk Nuxt 3

> TanStack Query adalah **satu-satunya** solusi untuk server state management. Pinia untuk UI state, TanStack Query untuk server state.

## Table of Contents
1. [Setup](#setup)
2. [Query Patterns](#query-patterns)
3. [Mutation Patterns](#mutation-patterns)
4. [Caching Strategies](#caching-strategies)
5. [Optimistic Updates](#optimistic-updates)
6. [Error Handling](#error-handling)
7. [SSR/Hydration](#ssrhydration)
8. [Advanced Patterns](#advanced-patterns)

---

## Setup

### Plugin Setup (Nuxt 3)

```typescript
// plugins/vue-query.ts
import { VueQueryPlugin, QueryClient, hydrate, dehydrate } from '@tanstack/vue-query'
import type { DehydratedState } from '@tanstack/vue-query'

export default defineNuxtPlugin((nuxtApp) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 1000 * 60 * 5, // 5 minutes
        gcTime: 1000 * 60 * 30,   // 30 minutes (formerly cacheTime)
        retry: 2,
        refetchOnWindowFocus: false, // Disable untuk UX yang lebih baik
        refetchOnReconnect: true,
      },
      mutations: {
        retry: 1,
      },
    },
  })

  nuxtApp.vueApp.use(VueQueryPlugin, { queryClient })

  // SSR Hydration
  if (import.meta.server) {
    nuxtApp.hooks.hook('app:rendered', () => {
      nuxtApp.payload.vueQueryState = dehydrate(queryClient)
    })
  }

  if (import.meta.client) {
    nuxtApp.hooks.hook('app:created', () => {
      const state = nuxtApp.payload.vueQueryState as DehydratedState
      if (state) {
        hydrate(queryClient, state)
      }
    })
  }

  return {
    provide: {
      queryClient,
    },
  }
})
```

### Query Keys (WAJIB Centralized)

```typescript
// composables/queries/keys.ts
export const queryKeys = {
  products: {
    all: ['products'] as const,
    lists: () => [...queryKeys.products.all, 'list'] as const,
    list: (filters: ProductFilters) => [...queryKeys.products.lists(), filters] as const,
    details: () => [...queryKeys.products.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.products.details(), id] as const,
  },
  users: {
    all: ['users'] as const,
    current: () => [...queryKeys.users.all, 'current'] as const,
    profile: (id: string) => [...queryKeys.users.all, 'profile', id] as const,
  },
  orders: {
    all: ['orders'] as const,
    list: (filters: OrderFilters) => [...queryKeys.orders.all, 'list', filters] as const,
    detail: (id: string) => [...queryKeys.orders.all, 'detail', id] as const,
  },
} as const

// Type helper untuk query keys
export type QueryKeys = typeof queryKeys
```

---

## Query Patterns

### Basic Query Composable

```typescript
// composables/queries/useProducts.ts
import { useQuery } from '@tanstack/vue-query'
import { queryKeys } from './keys'

interface ProductFilters {
  category?: string
  page?: number
  limit?: number
}

export function useProducts(filters: MaybeRef<ProductFilters> = {}) {
  const resolvedFilters = computed(() => toValue(filters))

  return useQuery({
    queryKey: computed(() => queryKeys.products.list(resolvedFilters.value)),
    queryFn: async () => {
      const params = new URLSearchParams()
      const f = resolvedFilters.value
      
      if (f.category) params.set('category', f.category)
      if (f.page) params.set('page', String(f.page))
      if (f.limit) params.set('limit', String(f.limit))
      
      const response = await $fetch<ProductsResponse>(`/api/products?${params}`)
      return response
    },
    // Hanya fetch jika ada filters yang valid
    enabled: computed(() => !!resolvedFilters.value),
  })
}
```

### Query dengan Dependent Data

```typescript
// composables/queries/useProductDetail.ts
export function useProductDetail(productId: MaybeRef<string | null>) {
  const id = computed(() => toValue(productId))

  return useQuery({
    queryKey: computed(() => queryKeys.products.detail(id.value!)),
    queryFn: async () => {
      const response = await $fetch<Product>(`/api/products/${id.value}`)
      return response
    },
    // PENTING: Disable query jika id belum ada
    enabled: computed(() => !!id.value),
  })
}

// Usage di component
const route = useRoute()
const productId = computed(() => route.params.id as string)
const { data: product, isLoading, error } = useProductDetail(productId)
```

### Infinite Query untuk Pagination

```typescript
// composables/queries/useInfiniteProducts.ts
import { useInfiniteQuery } from '@tanstack/vue-query'

interface ProductsPage {
  data: Product[]
  nextCursor: string | null
  hasMore: boolean
}

export function useInfiniteProducts(category: MaybeRef<string | null>) {
  const resolvedCategory = computed(() => toValue(category))

  return useInfiniteQuery({
    queryKey: computed(() => ['products', 'infinite', resolvedCategory.value]),
    queryFn: async ({ pageParam }) => {
      const params = new URLSearchParams()
      if (resolvedCategory.value) params.set('category', resolvedCategory.value)
      if (pageParam) params.set('cursor', pageParam)
      
      return await $fetch<ProductsPage>(`/api/products?${params}`)
    },
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.hasMore ? lastPage.nextCursor : undefined,
  })
}

// Usage
const { data, fetchNextPage, hasNextPage, isFetchingNextPage } = useInfiniteProducts(category)

const allProducts = computed(() => {
  return data.value?.pages.flatMap(page => page.data) ?? []
})
```

---

## Mutation Patterns

### Basic Mutation

```typescript
// composables/queries/useCreateProduct.ts
import { useMutation, useQueryClient } from '@tanstack/vue-query'

interface CreateProductInput {
  name: string
  price: number
  category: string
}

export function useCreateProduct() {
  const queryClient = useQueryClient()
  const toast = useToast()

  return useMutation({
    mutationFn: async (input: CreateProductInput) => {
      return await $fetch<Product>('/api/products', {
        method: 'POST',
        body: input,
      })
    },
    onSuccess: (newProduct) => {
      // Invalidate list queries
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.products.lists() 
      })
      
      // Optionally: Add to cache immediately
      queryClient.setQueryData(
        queryKeys.products.detail(newProduct.id),
        newProduct
      )
      
      toast.success('Product created successfully')
    },
    onError: (error) => {
      toast.error(`Failed to create product: ${error.message}`)
    },
  })
}

// Usage
const { mutate: createProduct, isPending } = useCreateProduct()

function handleSubmit(data: CreateProductInput) {
  createProduct(data)
}
```

### Mutation dengan Callback Options

```typescript
// Pattern untuk flexibility di component level
export function useUpdateProduct() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ id, data }: { id: string; data: UpdateProductInput }) => {
      return await $fetch<Product>(`/api/products/${id}`, {
        method: 'PUT',
        body: data,
      })
    },
    onSuccess: (updatedProduct) => {
      // Update detail cache
      queryClient.setQueryData(
        queryKeys.products.detail(updatedProduct.id),
        updatedProduct
      )
      // Invalidate lists
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.products.lists() 
      })
    },
  })
}

// Usage dengan per-call callbacks
const { mutate } = useUpdateProduct()

mutate(
  { id: productId, data: formData },
  {
    onSuccess: () => {
      // Component-specific logic
      navigateTo('/products')
    },
    onError: (error) => {
      // Handle specific error
      if (error.statusCode === 409) {
        showConflictDialog()
      }
    },
  }
)
```

---

## Caching Strategies

### Stale-While-Revalidate Pattern

```typescript
// Untuk data yang jarang berubah
const { data } = useQuery({
  queryKey: ['settings'],
  queryFn: fetchSettings,
  staleTime: 1000 * 60 * 30,  // 30 menit: data dianggap fresh
  gcTime: 1000 * 60 * 60,     // 1 jam: simpan di cache
})
```

### Real-time Data Pattern

```typescript
// Untuk data yang sering berubah (stock, notifications)
const { data } = useQuery({
  queryKey: ['stock', productId],
  queryFn: fetchStock,
  staleTime: 0,              // Selalu stale
  refetchInterval: 5000,     // Poll setiap 5 detik
  refetchIntervalInBackground: false, // Stop polling saat tab inactive
})
```

### Prefetching Pattern

```typescript
// composables/queries/useProductPrefetch.ts
export function useProductPrefetch() {
  const queryClient = useQueryClient()

  const prefetchProduct = async (productId: string) => {
    await queryClient.prefetchQuery({
      queryKey: queryKeys.products.detail(productId),
      queryFn: () => $fetch(`/api/products/${productId}`),
      staleTime: 1000 * 60 * 5, // 5 minutes
    })
  }

  return { prefetchProduct }
}

// Usage: Prefetch on hover
<template>
  <ProductCard 
    v-for="product in products"
    :key="product.id"
    :product="product"
    @mouseenter="prefetchProduct(product.id)"
  />
</template>
```

### Cache Seeding dari List ke Detail

```typescript
// Saat fetch list, seed cache untuk detail pages
export function useProducts(filters: MaybeRef<ProductFilters>) {
  const queryClient = useQueryClient()

  return useQuery({
    queryKey: computed(() => queryKeys.products.list(toValue(filters))),
    queryFn: async () => {
      const response = await $fetch<ProductsResponse>('/api/products')
      
      // Seed individual product caches
      response.data.forEach(product => {
        queryClient.setQueryData(
          queryKeys.products.detail(product.id),
          product
        )
      })
      
      return response
    },
  })
}
```

---

## Optimistic Updates

### Pattern 1: Simple Optimistic Update

```typescript
export function useToggleFavorite() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ productId, isFavorite }: { productId: string; isFavorite: boolean }) => {
      return await $fetch(`/api/products/${productId}/favorite`, {
        method: 'POST',
        body: { isFavorite },
      })
    },
    onMutate: async ({ productId, isFavorite }) => {
      // Cancel in-flight queries
      await queryClient.cancelQueries({ 
        queryKey: queryKeys.products.detail(productId) 
      })

      // Snapshot previous value
      const previousProduct = queryClient.getQueryData<Product>(
        queryKeys.products.detail(productId)
      )

      // Optimistically update
      if (previousProduct) {
        queryClient.setQueryData(
          queryKeys.products.detail(productId),
          { ...previousProduct, isFavorite }
        )
      }

      // Return context for rollback
      return { previousProduct }
    },
    onError: (err, variables, context) => {
      // Rollback on error
      if (context?.previousProduct) {
        queryClient.setQueryData(
          queryKeys.products.detail(variables.productId),
          context.previousProduct
        )
      }
    },
    onSettled: (data, error, variables) => {
      // Always refetch to ensure consistency
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.products.detail(variables.productId) 
      })
    },
  })
}
```

### Pattern 2: Optimistic List Update

```typescript
export function useDeleteProduct() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (productId: string) => {
      return await $fetch(`/api/products/${productId}`, { method: 'DELETE' })
    },
    onMutate: async (productId) => {
      await queryClient.cancelQueries({ queryKey: queryKeys.products.lists() })

      // Snapshot all list queries
      const previousLists = queryClient.getQueriesData<ProductsResponse>({
        queryKey: queryKeys.products.lists(),
      })

      // Optimistically remove from all lists
      queryClient.setQueriesData<ProductsResponse>(
        { queryKey: queryKeys.products.lists() },
        (old) => {
          if (!old) return old
          return {
            ...old,
            data: old.data.filter(p => p.id !== productId),
          }
        }
      )

      return { previousLists }
    },
    onError: (err, productId, context) => {
      // Rollback all list queries
      context?.previousLists.forEach(([queryKey, data]) => {
        queryClient.setQueryData(queryKey, data)
      })
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.products.lists() })
    },
  })
}
```

---

## Error Handling

### Global Error Handler

```typescript
// plugins/vue-query.ts
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors
        if (error?.statusCode >= 400 && error?.statusCode < 500) {
          return false
        }
        return failureCount < 2
      },
    },
    mutations: {
      onError: (error: any) => {
        // Global mutation error handler
        console.error('Mutation error:', error)
      },
    },
  },
  queryCache: new QueryCache({
    onError: (error, query) => {
      // Only show toast for background refetches that fail
      if (query.state.data !== undefined) {
        useToast().error(`Background update failed: ${error.message}`)
      }
    },
  }),
})
```

### Per-Query Error Handling

```typescript
// composables/queries/useProducts.ts
export function useProducts(filters: MaybeRef<ProductFilters>) {
  const toast = useToast()

  return useQuery({
    queryKey: computed(() => queryKeys.products.list(toValue(filters))),
    queryFn: async () => {
      try {
        return await $fetch<ProductsResponse>('/api/products')
      } catch (error: any) {
        // Transform error for better UX
        if (error.statusCode === 404) {
          return { data: [], total: 0 } // Return empty instead of error
        }
        throw error
      }
    },
    // Handle error state in component
    throwOnError: false,
  })
}

// Di component
const { data, error, isError } = useProducts(filters)

// Template
<template>
  <ErrorDisplay v-if="isError" :error="error" @retry="refetch" />
  <ProductList v-else :products="data?.data ?? []" />
</template>
```

---

## SSR/Hydration

### Proper SSR Setup

```typescript
// composables/queries/useProducts.ts
export function useProducts(filters: MaybeRef<ProductFilters>) {
  return useQuery({
    queryKey: computed(() => queryKeys.products.list(toValue(filters))),
    queryFn: async () => {
      // $fetch works on both server and client
      return await $fetch<ProductsResponse>('/api/products')
    },
  })
}

// pages/products/index.vue
<script setup lang="ts">
// Suspense akan menunggu query selesai di server
const { data, suspense } = useProducts({})
await suspense()
</script>
```

### Avoiding Hydration Mismatch

```typescript
// ❌ JANGAN: Client-only computed di SSR query
const { data } = useQuery({
  queryKey: ['products'],
  queryFn: async () => {
    const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone // Browser only!
    return await $fetch(`/api/products?tz=${userTimezone}`)
  },
})

// ✅ LAKUKAN: Handle client-only logic separately
const { data: rawData } = useQuery({
  queryKey: ['products'],
  queryFn: () => $fetch<ProductsResponse>('/api/products'),
})

const products = computed(() => {
  if (!rawData.value) return []
  return rawData.value.data.map(p => ({
    ...p,
    // Format di client side
    displayDate: formatDate(p.createdAt),
  }))
})
```

---

## Advanced Patterns

### Parallel Queries

```typescript
// composables/queries/useDashboardData.ts
export function useDashboardData() {
  const results = useQueries({
    queries: [
      {
        queryKey: queryKeys.products.lists(),
        queryFn: () => $fetch('/api/products'),
      },
      {
        queryKey: queryKeys.orders.list({}),
        queryFn: () => $fetch('/api/orders'),
      },
      {
        queryKey: ['stats'],
        queryFn: () => $fetch('/api/stats'),
      },
    ],
  })

  const isLoading = computed(() => results.some(r => r.isLoading))
  const isError = computed(() => results.some(r => r.isError))

  return {
    products: computed(() => results[0].data),
    orders: computed(() => results[1].data),
    stats: computed(() => results[2].data),
    isLoading,
    isError,
  }
}
```

### Mutation dengan Queue

```typescript
// Untuk mutations yang harus sequential
import { useMutationState } from '@tanstack/vue-query'

export function useOrderedMutations() {
  const pendingMutations = useMutationState({
    filters: { status: 'pending' },
  })

  const hasPendingMutations = computed(() => pendingMutations.value.length > 0)

  return {
    hasPendingMutations,
    pendingCount: computed(() => pendingMutations.value.length),
  }
}
```

### Query with Placeholders

```typescript
// Menampilkan skeleton dengan struktur data yang benar
export function useProducts(filters: MaybeRef<ProductFilters>) {
  return useQuery({
    queryKey: computed(() => queryKeys.products.list(toValue(filters))),
    queryFn: () => $fetch<ProductsResponse>('/api/products'),
    placeholderData: {
      data: Array(10).fill(null).map((_, i) => ({
        id: `placeholder-${i}`,
        name: '',
        price: 0,
        isPlaceholder: true,
      })),
      total: 10,
    },
  })
}

// Template dengan skeleton
<template>
  <ProductCard 
    v-for="product in data?.data"
    :key="product.id"
    :product="product"
    :class="{ 'animate-pulse': product.isPlaceholder }"
  />
</template>
```
