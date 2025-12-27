# Clean Code untuk Vue.js & Nuxt 3

> "Any fool can write code that a computer can understand. Good programmers write code that humans can understand." — Martin Fowler

## Table of Contents
1. [Naming Conventions](#naming-conventions)
2. [Component Design](#component-design)
3. [Composables Patterns](#composables-patterns)
4. [TypeScript Best Practices](#typescript-best-practices)
5. [Code Smells & Refactoring](#code-smells--refactoring)
6. [Documentation Standards](#documentation-standards)

---

## Naming Conventions

### Variables & Functions

```typescript
// ❌ BAD: Cryptic names
const d = new Date()
const arr = []
const obj = {}
const fn = () => {}

// ✅ GOOD: Descriptive names
const currentDate = new Date()
const activeProducts = []
const userSettings = {}
const calculateTotalPrice = () => {}
```

### Boolean Naming

```typescript
// ❌ BAD: Ambiguous
const open = ref(false)
const data = ref(false)
const click = ref(false)

// ✅ GOOD: Prefixed dengan is/has/can/should
const isOpen = ref(false)
const hasData = ref(false)
const isClickable = ref(false)
const shouldAutoSave = ref(true)
const canSubmit = computed(() => isValid.value && !isLoading.value)
```

### Event Handlers

```typescript
// ❌ BAD: Vague names
function click() {}
function change() {}
function submit() {}

// ✅ GOOD: Action-oriented dengan handle prefix
function handleButtonClick() {}
function handleInputChange(value: string) {}
function handleFormSubmit() {}

// Atau dengan on prefix untuk event emissions
const emit = defineEmits<{
  'update:modelValue': [value: string]
  'submit': [data: FormData]
  'close': []
}>()
```

### Composables Naming

```typescript
// ✅ SELALU prefix dengan 'use'
export function useProducts() {}      // Data fetching
export function useAuth() {}          // Auth state
export function useScrollLock() {}    // Side effect
export function useLocalStorage() {}  // Browser API wrapper
export function useDebounce() {}      // Utility

// Return object dengan clear names
export function useProducts() {
  return {
    products,           // Data
    isLoading,          // State
    error,              // Error
    fetchProducts,      // Action
    refetch,            // Action
  }
}
```

### Components Naming

```typescript
// ❌ BAD: Generic or unclear
Button.vue
Modal.vue
Card.vue
Item.vue
List.vue

// ✅ GOOD: Specific and prefixed
BaseButton.vue        // Base components
BaseModal.vue
BaseCard.vue

ProductCard.vue       // Feature components
ProductList.vue
ProductListItem.vue

TheHeader.vue         // Singleton components (hanya 1 di app)
TheSidebar.vue
TheFooter.vue

AppNavigation.vue     // App-level components
AppBreadcrumb.vue
```

---

## Component Design

### Single Responsibility

```vue
<!-- ❌ BAD: Component yang melakukan terlalu banyak -->
<script setup lang="ts">
const { data } = await useFetch('/api/products')
const { data: categories } = await useFetch('/api/categories')
const cart = useCartStore()
const filters = ref({ category: '', minPrice: 0, maxPrice: 100 })
const sortBy = ref('name')

const filteredProducts = computed(() => { /* 50 lines */ })
const sortedProducts = computed(() => { /* 20 lines */ })

function addToCart(product: Product) { /* 30 lines */ }
function handleFilter(newFilters: Filters) { /* 20 lines */ }
</script>

<template>
  <!-- 200 lines of template -->
</template>
```

```vue
<!-- ✅ GOOD: Dipecah menjadi components dan composables -->

<!-- ProductsPage.vue - Orchestration only -->
<script setup lang="ts">
const filters = ref<ProductFilters>({})
</script>

<template>
  <div class="products-page">
    <ProductFilters v-model="filters" />
    <ProductList :filters="filters" />
  </div>
</template>

<!-- ProductList.vue - Display logic -->
<script setup lang="ts">
interface Props {
  filters: ProductFilters
}
const props = defineProps<Props>()
const { products, isLoading } = useProducts(toRef(props, 'filters'))
</script>

<template>
  <BaseSpinner v-if="isLoading" />
  <div v-else class="product-grid">
    <ProductCard 
      v-for="product in products" 
      :key="product.id" 
      :product="product" 
    />
  </div>
</template>
```

### Props Design

```typescript
// ❌ BAD: Terlalu banyak props
interface Props {
  title: string
  subtitle: string
  description: string
  image: string
  imageAlt: string
  buttonText: string
  buttonVariant: string
  showButton: boolean
  onClick: () => void
  // ... 10 more props
}

// ✅ GOOD: Grouped props atau gunakan slots
interface Props {
  product: Product
  showActions?: boolean
}

// Template menggunakan product object
<template>
  <article class="product-card">
    <img :src="product.image" :alt="product.name">
    <h3>{{ product.name }}</h3>
    <p>{{ product.description }}</p>
    <slot name="actions" v-if="showActions" />
  </article>
</template>
```

### Emit Design

```typescript
// ❌ BAD: Emit dengan any atau tanpa type
const emit = defineEmits(['update', 'change', 'click'])

// ✅ GOOD: Typed emits dengan payload types
const emit = defineEmits<{
  'update:modelValue': [value: string]
  'select': [item: Product]
  'delete': [id: string]
  'submit': [data: FormData, options?: SubmitOptions]
}>()
```

### Slots untuk Flexibility

```vue
<!-- ❌ BAD: Props untuk semua customization -->
<DataTable
  :data="users"
  :columns="columns"
  :cellRenderer="renderCell"
  :headerRenderer="renderHeader"
  :emptyText="'No data'"
  :loadingComponent="CustomSpinner"
/>

<!-- ✅ GOOD: Slots untuk customization -->
<DataTable :data="users" :columns="columns">
  <template #header="{ column }">
    <span class="custom-header">{{ column.label }}</span>
  </template>
  
  <template #cell-status="{ value }">
    <StatusBadge :status="value" />
  </template>
  
  <template #empty>
    <EmptyState message="No users found" />
  </template>
  
  <template #loading>
    <TableSkeleton :rows="5" />
  </template>
</DataTable>
```

---

## Composables Patterns

### Separation of Concerns

```typescript
// ❌ BAD: Monolithic composable
export function useProducts() {
  // State
  const products = ref<Product[]>([])
  const loading = ref(false)
  const error = ref<Error | null>(null)
  const filters = ref<ProductFilters>({})
  const sortBy = ref('name')
  const cart = ref<CartItem[]>([])
  
  // Fetch logic
  async function fetchProducts() { /* ... */ }
  
  // Filter logic
  const filteredProducts = computed(() => { /* ... */ })
  
  // Sort logic
  const sortedProducts = computed(() => { /* ... */ })
  
  // Cart logic
  function addToCart() { /* ... */ }
  function removeFromCart() { /* ... */ }
  
  // 200+ more lines...
}
```

```typescript
// ✅ GOOD: Separated composables
// composables/queries/useProducts.ts
export function useProducts(filters: MaybeRef<ProductFilters>) {
  return useQuery({
    queryKey: computed(() => ['products', toValue(filters)]),
    queryFn: () => fetchProducts(toValue(filters)),
  })
}

// composables/useProductFilters.ts
export function useProductFilters(initialFilters: ProductFilters = {}) {
  const filters = ref<ProductFilters>(initialFilters)
  
  function setCategory(category: string) {
    filters.value.category = category
  }
  
  function resetFilters() {
    filters.value = initialFilters
  }
  
  return {
    filters: readonly(filters),
    setCategory,
    resetFilters,
  }
}

// composables/useProductSort.ts
export function useProductSort<T extends { [key: string]: any }>(
  items: Ref<T[]>,
  defaultSort: keyof T = 'name'
) {
  const sortBy = ref<keyof T>(defaultSort)
  const sortOrder = ref<'asc' | 'desc'>('asc')
  
  const sorted = computed(() => {
    return [...items.value].sort((a, b) => {
      const modifier = sortOrder.value === 'asc' ? 1 : -1
      return a[sortBy.value] > b[sortBy.value] ? modifier : -modifier
    })
  })
  
  return { sorted, sortBy, sortOrder }
}
```

### Composable Return Pattern

```typescript
// ✅ GOOD: Consistent return structure
export function useFeature() {
  // State
  const data = ref<Data | null>(null)
  const isLoading = ref(false)
  const error = ref<Error | null>(null)
  
  // Computed
  const isEmpty = computed(() => !data.value || data.value.length === 0)
  const hasError = computed(() => !!error.value)
  
  // Actions
  async function fetch() { /* ... */ }
  function reset() { /* ... */ }
  
  // Return dengan KONSISTEN grouping:
  // 1. Data (reactive)
  // 2. State indicators
  // 3. Computed values
  // 4. Actions
  return {
    // Data - use readonly untuk prevent external mutation
    data: readonly(data),
    
    // State
    isLoading: readonly(isLoading),
    error: readonly(error),
    
    // Computed
    isEmpty,
    hasError,
    
    // Actions
    fetch,
    reset,
  }
}
```

### Composable dengan Options

```typescript
// ✅ GOOD: Options pattern untuk flexibility
interface UseDebounceOptions {
  delay?: number
  immediate?: boolean
  maxWait?: number
}

export function useDebounce<T>(
  value: Ref<T>,
  options: UseDebounceOptions = {}
) {
  const {
    delay = 300,
    immediate = false,
    maxWait,
  } = options
  
  const debouncedValue = ref(value.value) as Ref<T>
  
  const debouncedFn = useDebounceFn(
    () => {
      debouncedValue.value = value.value
    },
    delay,
    { maxWait }
  )
  
  watch(value, () => {
    if (immediate) {
      debouncedValue.value = value.value
    }
    debouncedFn()
  })
  
  return debouncedValue
}

// Usage
const searchQuery = ref('')
const debouncedQuery = useDebounce(searchQuery, { delay: 500 })
```

---

## TypeScript Best Practices

### Type Definition

```typescript
// ❌ BAD: Type di mana-mana
// components/ProductCard.vue
interface Product {
  id: string
  name: string
  price: number
}

// pages/products/index.vue
interface Product {
  id: string
  name: string
  price: number
  category: string  // Different!
}

// ✅ GOOD: Centralized types
// types/models.ts
export interface Product {
  id: string
  name: string
  price: number
  category: string
  description?: string
  createdAt: Date
  updatedAt: Date
}

// Untuk partial updates
export type ProductCreate = Omit<Product, 'id' | 'createdAt' | 'updatedAt'>
export type ProductUpdate = Partial<ProductCreate>

// Untuk API responses
export interface ProductsResponse {
  data: Product[]
  meta: {
    total: number
    page: number
    limit: number
  }
}
```

### Strict Types - No `any`

```typescript
// ❌ BAD: any everywhere
async function fetchData(): Promise<any> {
  const response: any = await $fetch('/api/data')
  return response.data
}

function processItems(items: any[]) {
  return items.map((item: any) => item.value)
}

// ✅ GOOD: Proper typing
interface ApiResponse<T> {
  data: T
  status: number
  message?: string
}

async function fetchData<T>(): Promise<T> {
  const response = await $fetch<ApiResponse<T>>('/api/data')
  return response.data
}

interface Item {
  id: string
  value: number
}

function processItems(items: Item[]): number[] {
  return items.map(item => item.value)
}
```

### Generic Composables

```typescript
// ✅ GOOD: Generic untuk reusability
export function useSelection<T extends { id: string }>(items: Ref<T[]>) {
  const selectedIds = ref<Set<string>>(new Set())
  
  const selectedItems = computed(() => 
    items.value.filter(item => selectedIds.value.has(item.id))
  )
  
  const isSelected = (item: T) => selectedIds.value.has(item.id)
  
  function toggle(item: T) {
    if (selectedIds.value.has(item.id)) {
      selectedIds.value.delete(item.id)
    } else {
      selectedIds.value.add(item.id)
    }
  }
  
  function selectAll() {
    items.value.forEach(item => selectedIds.value.add(item.id))
  }
  
  function clearSelection() {
    selectedIds.value.clear()
  }
  
  return {
    selectedIds: readonly(selectedIds),
    selectedItems,
    isSelected,
    toggle,
    selectAll,
    clearSelection,
  }
}

// Usage
const products = ref<Product[]>([])
const { selectedItems, toggle, selectAll } = useSelection(products)
```

---

## Code Smells & Refactoring

### Smell: Long Parameter List

```typescript
// ❌ BAD
function createProduct(
  name: string,
  price: number,
  category: string,
  description: string,
  image: string,
  stock: number,
  isActive: boolean,
  tags: string[]
) { /* ... */ }

// ✅ GOOD: Object parameter
interface CreateProductParams {
  name: string
  price: number
  category: string
  description?: string
  image?: string
  stock?: number
  isActive?: boolean
  tags?: string[]
}

function createProduct(params: CreateProductParams) { /* ... */ }

// Usage lebih clear
createProduct({
  name: 'iPhone',
  price: 999,
  category: 'electronics',
})
```

### Smell: Nested Conditionals

```typescript
// ❌ BAD: Deeply nested
function processOrder(order: Order) {
  if (order) {
    if (order.items.length > 0) {
      if (order.status === 'pending') {
        if (order.payment) {
          if (order.payment.verified) {
            // Finally do something
          }
        }
      }
    }
  }
}

// ✅ GOOD: Early returns (Guard clauses)
function processOrder(order: Order) {
  if (!order) return
  if (order.items.length === 0) return
  if (order.status !== 'pending') return
  if (!order.payment?.verified) return
  
  // Clean main logic
  fulfillOrder(order)
}
```

### Smell: Magic Numbers/Strings

```typescript
// ❌ BAD
if (user.role === 'admin') { /* ... */ }
if (items.length > 10) { /* ... */ }
const timeout = 5000

// ✅ GOOD: Named constants
// utils/constants.ts
export const USER_ROLES = {
  ADMIN: 'admin',
  USER: 'user',
  GUEST: 'guest',
} as const

export const LIMITS = {
  MAX_ITEMS_PER_PAGE: 10,
  MAX_CART_ITEMS: 50,
  API_TIMEOUT_MS: 5000,
} as const

// Usage
import { USER_ROLES, LIMITS } from '~/utils/constants'

if (user.role === USER_ROLES.ADMIN) { /* ... */ }
if (items.length > LIMITS.MAX_ITEMS_PER_PAGE) { /* ... */ }
```

### Smell: Repeated Patterns

```typescript
// ❌ BAD: Copy-paste error handling
async function fetchProducts() {
  try {
    isLoading.value = true
    const data = await $fetch('/api/products')
    return data
  } catch (e) {
    error.value = e as Error
    toast.error('Failed to fetch products')
  } finally {
    isLoading.value = false
  }
}

async function fetchUsers() {
  try {
    isLoading.value = true
    const data = await $fetch('/api/users')
    return data
  } catch (e) {
    error.value = e as Error
    toast.error('Failed to fetch users')
  } finally {
    isLoading.value = false
  }
}

// ✅ GOOD: Extract pattern (atau gunakan TanStack Query!)
function useAsyncAction<T>(
  action: () => Promise<T>,
  errorMessage: string
) {
  const isLoading = ref(false)
  const error = ref<Error | null>(null)
  const data = ref<T | null>(null)
  
  async function execute() {
    try {
      isLoading.value = true
      error.value = null
      data.value = await action()
      return data.value
    } catch (e) {
      error.value = e as Error
      toast.error(errorMessage)
      throw e
    } finally {
      isLoading.value = false
    }
  }
  
  return { data, isLoading, error, execute }
}
```

---

## Documentation Standards

### Component Documentation

```vue
<script setup lang="ts">
/**
 * ProductCard - Menampilkan informasi produk dalam card format
 * 
 * @example
 * ```vue
 * <ProductCard 
 *   :product="product" 
 *   show-actions
 *   @add-to-cart="handleAddToCart"
 * />
 * ```
 */

interface Props {
  /** Data produk yang akan ditampilkan */
  product: Product
  /** Tampilkan action buttons */
  showActions?: boolean
  /** Disable semua interactions */
  disabled?: boolean
}

interface Emits {
  /** Triggered ketika user klik "Add to Cart" */
  (e: 'addToCart', product: Product): void
  /** Triggered ketika user klik card */
  (e: 'select', product: Product): void
}

const props = withDefaults(defineProps<Props>(), {
  showActions: true,
  disabled: false,
})

const emit = defineEmits<Emits>()
</script>
```

### Composable Documentation

```typescript
/**
 * useProducts - Composable untuk fetching dan managing products
 * 
 * Menggunakan TanStack Query untuk caching dan background updates.
 * 
 * @param filters - Reactive filters untuk query
 * @returns Object dengan products data, loading state, dan actions
 * 
 * @example
 * ```typescript
 * const filters = ref({ category: 'electronics' })
 * const { products, isLoading, refetch } = useProducts(filters)
 * ```
 */
export function useProducts(filters: MaybeRef<ProductFilters>) {
  // Implementation
}
```

### Function Documentation

```typescript
/**
 * Formats a price value to Indonesian Rupiah format
 * 
 * @param price - The price value to format
 * @param options - Formatting options
 * @returns Formatted price string
 * 
 * @example
 * ```typescript
 * formatPrice(50000) // "Rp 50.000"
 * formatPrice(1500000, { compact: true }) // "Rp 1,5jt"
 * ```
 */
export function formatPrice(
  price: number,
  options: FormatPriceOptions = {}
): string {
  // Implementation
}
```
