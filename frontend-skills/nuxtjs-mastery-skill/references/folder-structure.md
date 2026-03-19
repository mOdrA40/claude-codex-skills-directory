# Folder Structure — Production-Ready Nuxt 3

> Filosofi: **Feature-based organization** — Code yang related harus hidup berdekatan.

## Table of Contents
1. [Struktur Lengkap](#struktur-lengkap)
2. [Penjelasan Setiap Direktori](#penjelasan-setiap-direktori)
3. [Naming Conventions](#naming-conventions)
4. [Anti-patterns](#anti-patterns)

---

## Struktur Lengkap

```
my-nuxt-app/
├── .nuxt/                    # Auto-generated (gitignore)
├── .output/                  # Build output (gitignore)
├── assets/
│   ├── css/
│   │   ├── main.css          # Global styles (minimal!)
│   │   └── variables.css     # CSS custom properties
│   ├── fonts/
│   └── images/               # Static images yang perlu processing
├── components/
│   ├── base/                 # Reusable primitives
│   │   ├── BaseButton.vue
│   │   ├── BaseInput.vue
│   │   ├── BaseModal.vue
│   │   └── BaseSpinner.vue
│   ├── common/               # Shared across features
│   │   ├── AppHeader.vue
│   │   ├── AppFooter.vue
│   │   ├── AppSidebar.vue
│   │   └── ErrorDisplay.vue
│   └── features/             # Feature-specific components
│       ├── auth/
│       │   ├── LoginForm.vue
│       │   └── RegisterForm.vue
│       ├── products/
│       │   ├── ProductCard.vue
│       │   ├── ProductList.vue
│       │   └── ProductFilters.vue
│       └── cart/
│           ├── CartItem.vue
│           └── CartSummary.vue
├── composables/              # Shared logic (auto-imported)
│   ├── useApi.ts             # API wrapper
│   ├── useAuth.ts            # Auth state & methods
│   ├── useToast.ts           # Toast notifications
│   └── queries/              # TanStack Query hooks
│       ├── useProducts.ts
│       ├── useUser.ts
│       └── useOrders.ts
├── layouts/
│   ├── default.vue           # Main layout
│   ├── auth.vue              # Login/register pages
│   └── dashboard.vue         # Admin area
├── middleware/
│   ├── auth.ts               # Protected routes
│   └── guest.ts              # Redirect if logged in
├── pages/
│   ├── index.vue             # Homepage
│   ├── login.vue
│   ├── register.vue
│   ├── products/
│   │   ├── index.vue         # /products
│   │   └── [id].vue          # /products/:id
│   └── dashboard/
│       ├── index.vue         # /dashboard
│       └── settings.vue      # /dashboard/settings
├── plugins/
│   ├── vue-query.ts          # TanStack Query setup
│   └── api.ts                # $api plugin
├── public/                   # Static assets (no processing)
│   ├── favicon.ico
│   └── robots.txt
├── server/
│   ├── api/                  # API routes
│   │   ├── auth/
│   │   │   ├── login.post.ts
│   │   │   └── logout.post.ts
│   │   └── products/
│   │       ├── index.get.ts
│   │       └── [id].get.ts
│   ├── middleware/           # Server middleware
│   │   └── log.ts
│   └── utils/                # Server utilities
│       ├── db.ts
│       └── auth.ts
├── stores/                   # Pinia stores
│   ├── auth.ts
│   └── ui.ts                 # UI state (sidebar, theme)
├── types/
│   ├── index.ts              # Re-export semua types
│   ├── api.ts                # API response types
│   ├── models.ts             # Domain models
│   └── components.ts         # Component props types
├── utils/
│   ├── constants.ts          # App constants
│   ├── helpers.ts            # Pure utility functions
│   ├── validators.ts         # Zod schemas
│   └── formatters.ts         # Date, currency formatters
├── .env                      # Environment variables
├── .env.example              # Template for .env
├── app.vue                   # Root component
├── nuxt.config.ts
├── package.json
└── tsconfig.json
```

---

## Penjelasan Setiap Direktori

### `/components`

**Struktur 3-level yang WAJIB diikuti:**

```
components/
├── base/      → Primitives tanpa business logic
├── common/    → Shared dengan sedikit business logic  
└── features/  → Feature-specific, boleh complex
```

**Rules:**
- `base/` — Tidak boleh import dari `features/` atau `common/`
- `common/` — Boleh import dari `base/`, tidak boleh dari `features/`
- `features/` — Boleh import dari mana saja

**Contoh BaseButton yang benar:**

```vue
<!-- components/base/BaseButton.vue -->
<script setup lang="ts">
interface Props {
  variant?: 'primary' | 'secondary' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
  disabled?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
  loading: false,
  disabled: false
})

const emit = defineEmits<{
  click: [event: MouseEvent]
}>()
</script>

<template>
  <button
    :class="[
      'btn',
      `btn--${variant}`,
      `btn--${size}`,
      { 'btn--loading': loading }
    ]"
    :disabled="disabled || loading"
    @click="emit('click', $event)"
  >
    <BaseSpinner v-if="loading" size="sm" />
    <slot v-else />
  </button>
</template>
```

### `/composables`

**Struktur yang scalable:**

```
composables/
├── useApi.ts           # Low-level API wrapper
├── useAuth.ts          # Auth composable
├── useToast.ts         # UI feedback
└── queries/            # TanStack Query hooks (PENTING!)
    ├── index.ts        # Re-export semua
    ├── useProducts.ts
    ├── useUser.ts
    └── keys.ts         # Query keys centralized
```

**Query keys WAJIB centralized:**

```typescript
// composables/queries/keys.ts
export const queryKeys = {
  products: {
    all: ['products'] as const,
    list: (filters: ProductFilters) => ['products', 'list', filters] as const,
    detail: (id: string) => ['products', 'detail', id] as const,
  },
  users: {
    current: ['users', 'current'] as const,
    profile: (id: string) => ['users', 'profile', id] as const,
  },
} as const
```

### `/pages`

**Halaman harus MINIMAL — hanya orchestration:**

```vue
<!-- pages/products/index.vue -->
<script setup lang="ts">
// ✅ Page hanya orchestrate components
definePageMeta({
  middleware: ['auth']
})

const route = useRoute()
const filters = computed(() => ({
  category: route.query.category as string,
  page: Number(route.query.page) || 1
}))
</script>

<template>
  <div class="products-page">
    <ProductFilters v-model="filters" />
    <ProductList :filters="filters" />
  </div>
</template>
```

### `/stores` (Pinia)

**Gunakan HANYA untuk:**
- UI state (sidebar open, theme, etc.)
- Auth state yang perlu persist
- State yang diakses > 3 components tanpa parent-child relation

**JANGAN gunakan untuk:**
- Server state (gunakan TanStack Query!)
- Form state (gunakan VeeValidate!)
- Component-local state

```typescript
// stores/ui.ts
export const useUiStore = defineStore('ui', () => {
  const sidebarOpen = ref(false)
  const theme = ref<'light' | 'dark'>('light')

  const toggleSidebar = () => {
    sidebarOpen.value = !sidebarOpen.value
  }

  return {
    sidebarOpen: readonly(sidebarOpen),
    theme: readonly(theme),
    toggleSidebar,
  }
})
```

### `/server`

**File naming convention untuk API routes:**

```
server/api/
├── products/
│   ├── index.get.ts      # GET /api/products
│   ├── index.post.ts     # POST /api/products
│   └── [id].get.ts       # GET /api/products/:id
│   └── [id].put.ts       # PUT /api/products/:id
│   └── [id].delete.ts    # DELETE /api/products/:id
```

**Server API handler pattern:**

```typescript
// server/api/products/index.get.ts
import { z } from 'zod'

const querySchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
  category: z.string().optional(),
})

export default defineEventHandler(async (event) => {
  const query = await getValidatedQuery(event, querySchema.parse)
  
  // Business logic here
  const products = await fetchProducts(query)
  
  return {
    data: products,
    meta: { page: query.page, limit: query.limit }
  }
})
```

---

## Naming Conventions

### Files

| Type | Convention | Example |
|------|------------|---------|
| Components | PascalCase | `ProductCard.vue` |
| Composables | camelCase dengan `use` prefix | `useProducts.ts` |
| Stores | camelCase | `auth.ts` |
| Utils | camelCase | `formatters.ts` |
| Types | camelCase | `models.ts` |
| Pages | kebab-case | `product-detail.vue` |

### Code

```typescript
// Variables & functions: camelCase
const productList = ref([])
function fetchProducts() {}

// Types & interfaces: PascalCase
interface ProductResponse {}
type ProductStatus = 'active' | 'inactive'

// Constants: SCREAMING_SNAKE_CASE
const MAX_ITEMS_PER_PAGE = 50
const API_BASE_URL = '/api'

// Enums: PascalCase with PascalCase members
enum OrderStatus {
  Pending = 'pending',
  Processing = 'processing',
  Completed = 'completed'
}
```

---

## Anti-patterns

### ❌ Component yang Terlalu Besar

```vue
<!-- JANGAN: 500+ lines component -->
<script setup>
// 200 lines of logic
</script>
<template>
  <!-- 300 lines of template -->
</template>
```

**Solusi:** Pecah jadi smaller components + composables.

### ❌ Business Logic di Pages

```vue
<!-- JANGAN -->
<script setup>
const { data } = await useFetch('/api/products')
const filtered = computed(() => {
  // 50 lines filtering logic
})
const sorted = computed(() => {
  // 30 lines sorting logic
})
</script>
```

**Solusi:** Extract ke composable.

### ❌ Deeply Nested Folders

```
components/
└── features/
    └── products/
        └── list/
            └── item/
                └── actions/
                    └── Button.vue  # TERLALU DALAM!
```

**Rule:** Maximum 3 levels deep.

### ❌ Barrel Files yang Berlebihan

```typescript
// JANGAN: index.ts di setiap folder
// components/base/index.ts
export * from './BaseButton.vue'
export * from './BaseInput.vue'
// ... dst

// Ini memperlambat HMR dan tree-shaking
```

**Solusi:** Nuxt auto-import handles this.
