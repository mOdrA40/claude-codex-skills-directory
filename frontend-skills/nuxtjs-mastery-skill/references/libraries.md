# Curated Libraries untuk Vue.js & Nuxt 3

> Library yang dipilih berdasarkan: maintenance aktif, TypeScript support, bundle size, community adoption, dan battle-tested di production.

## Table of Contents
1. [Core Essentials](#core-essentials)
2. [Data Fetching & State](#data-fetching--state)
3. [UI Components](#ui-components)
4. [Forms & Validation](#forms--validation)
5. [Utilities](#utilities)
6. [Animation & UX](#animation--ux)
7. [Testing](#testing)
8. [DevTools & DX](#devtools--dx)
9. [Libraries to AVOID](#libraries-to-avoid)

---

## Core Essentials

### TanStack Query (Vue Query)
**Server state management - WAJIB**

```bash
npm install @tanstack/vue-query
```

```typescript
// Kenapa WAJIB?
// - Caching otomatis
// - Background refetching
// - Optimistic updates
// - Request deduplication
// - Pagination, infinite scroll built-in

// Kapan pakai?
// SEMUA API calls. No exceptions.
```

### VueUse
**Collection of Vue composition utilities - WAJIB**

```bash
npm install @vueuse/core @vueuse/nuxt
```

```typescript
// 200+ composables yang production-ready
import {
  useLocalStorage,    // Reactive localStorage
  useDark,            // Dark mode toggle
  useDebounce,        // Debounced ref
  useEventListener,   // Auto-cleanup event listener
  useIntersectionObserver, // Lazy loading
  useVirtualList,     // Virtual scroll
  useFetch,           // Fetch wrapper (prefer TanStack Query)
} from '@vueuse/core'

// Kapan pakai?
// Browser APIs, utilities, side effects
```

### Pinia
**Store untuk UI/client state - Built into Nuxt**

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  modules: ['@pinia/nuxt'],
})

// Kapan pakai Pinia vs TanStack Query?
// Pinia: UI state (sidebar, theme, auth status)
// TanStack Query: Server state (API data)
```

---

## Data Fetching & State

### $fetch (Built-in Nuxt)
**HTTP client untuk Nuxt**

```typescript
// Gunakan untuk:
// - Server-side data fetching
// - API routes
// - Sebagai queryFn di TanStack Query

const data = await $fetch('/api/products', {
  method: 'POST',
  body: { name: 'Product' },
})
```

### ofetch
**Underlying library untuk $fetch**

```bash
npm install ofetch
```

```typescript
// Gunakan jika butuh custom instance
import { ofetch } from 'ofetch'

const api = ofetch.create({
  baseURL: 'https://api.example.com',
  headers: {
    Authorization: `Bearer ${token}`,
  },
})
```

---

## UI Components

### Radix Vue
**Unstyled, accessible components - RECOMMENDED**

```bash
npm install radix-vue
```

```typescript
// Kenapa Radix Vue?
// - Unstyled = full control
// - Accessible by default (WCAG 2.1)
// - Keyboard navigation built-in
// - Focus management

import { 
  DialogRoot, DialogTrigger, DialogContent,
  DropdownMenuRoot, DropdownMenuTrigger,
  TooltipRoot, TooltipTrigger, TooltipContent,
} from 'radix-vue'

// Kapan pakai?
// - Custom design system
// - Need full styling control
// - Accessibility is priority
```

### Headless UI
**Alternative untuk Radix**

```bash
npm install @headlessui/vue
```

```typescript
// Dari Tailwind Labs
// Mirip Radix, lebih opinionated

import { 
  Dialog, 
  Menu, 
  Listbox, 
  Combobox,
  Switch,
} from '@headlessui/vue'
```

### Nuxt UI
**Full component library dengan Tailwind**

```bash
npm install @nuxt/ui
```

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  modules: ['@nuxt/ui'],
})

// Kapan pakai?
// - Rapid prototyping
// - Tim kecil, butuh cepat
// - Tailwind-based project
```

### PrimeVue
**Enterprise-grade component library**

```bash
npm install primevue @primevue/themes
```

```typescript
// 80+ komponen lengkap
// Data tables, charts, forms, dll

// Kapan pakai?
// - Enterprise apps
// - Complex data tables needed
// - Full-featured admin dashboards
```

---

## Forms & Validation

### VeeValidate + Zod
**Form validation - RECOMMENDED COMBO**

```bash
npm install vee-validate @vee-validate/zod zod
```

```typescript
// composables/useProductForm.ts
import { useForm } from 'vee-validate'
import { toTypedSchema } from '@vee-validate/zod'
import { z } from 'zod'

const productSchema = z.object({
  name: z.string().min(3, 'Minimum 3 characters'),
  price: z.number().positive('Must be positive'),
  category: z.string().min(1, 'Required'),
})

export function useProductForm() {
  const { handleSubmit, errors, values } = useForm({
    validationSchema: toTypedSchema(productSchema),
    initialValues: {
      name: '',
      price: 0,
      category: '',
    },
  })

  return { handleSubmit, errors, values }
}
```

### Zod
**Schema validation - WAJIB**

```bash
npm install zod
```

```typescript
// Gunakan EVERYWHERE:
// - Form validation
// - API response validation
// - Environment variables
// - Config validation

import { z } from 'zod'

// API response validation
const ProductSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  price: z.number().positive(),
  createdAt: z.string().datetime(),
})

type Product = z.infer<typeof ProductSchema>

// Validate API response
const product = ProductSchema.parse(apiResponse)
```

### FormKit
**Alternative form library**

```bash
npm install @formkit/vue @formkit/nuxt
```

```typescript
// More opinionated, form-builder style
// Bagus untuk forms yang complex

// Kapan pakai?
// - Multi-step forms
// - Dynamic form generation
// - Form builder features needed
```

---

## Utilities

### date-fns
**Date manipulation - RECOMMENDED**

```bash
npm install date-fns
```

```typescript
// Kenapa date-fns?
// - Tree-shakeable (tidak bloat bundle)
// - Immutable (tidak mutate date)
// - TypeScript native

import { format, parseISO, addDays, isAfter } from 'date-fns'
import { id } from 'date-fns/locale' // Indonesian locale

format(new Date(), 'dd MMMM yyyy', { locale: id })
// "25 Desember 2024"
```

### nanoid
**ID generator - RECOMMENDED**

```bash
npm install nanoid
```

```typescript
// Kenapa nanoid vs uuid?
// - Lebih pendek (21 chars vs 36)
// - URL-safe by default
// - Lebih kecil (130 bytes vs 2.5KB)

import { nanoid } from 'nanoid'

const id = nanoid() // "V1StGXR8_Z5jdHi6B-myT"

// Custom length
const shortId = nanoid(10) // "IRFa-VaY2b"
```

### lodash-es
**Utility functions (jika butuh)**

```bash
npm install lodash-es
```

```typescript
// JANGAN import semua lodash!
// ❌ import _ from 'lodash' // 70KB!

// ✅ Import per function
import debounce from 'lodash-es/debounce'
import groupBy from 'lodash-es/groupBy'

// LEBIH BAIK: Gunakan VueUse atau native JS
// VueUse punya useDebounceFn, useThrottleFn
// Native: Array.prototype.groupBy (stage 3)
```

### ky
**HTTP client alternative**

```bash
npm install ky
```

```typescript
// Lebih modern dari axios
// Lebih kecil, promise-based, hooks

import ky from 'ky'

const api = ky.extend({
  prefixUrl: 'https://api.example.com',
  hooks: {
    beforeRequest: [(request) => {
      request.headers.set('Authorization', `Bearer ${token}`)
    }],
  },
})

// Tapi di Nuxt, prefer $fetch
```

---

## Animation & UX

### Motion One / @vueuse/motion
**Animation library - RECOMMENDED**

```bash
npm install @vueuse/motion
```

```typescript
// Nuxt module
// nuxt.config.ts
export default defineNuxtConfig({
  modules: ['@vueuse/motion/nuxt'],
})

// Usage
<template>
  <div v-motion-fade-visible>
    Fade in when visible
  </div>
  
  <div
    v-motion
    :initial="{ opacity: 0, y: 100 }"
    :enter="{ opacity: 1, y: 0 }"
  >
    Custom animation
  </div>
</template>
```

### Lottie
**Complex animations**

```bash
npm install lottie-web vue3-lottie
```

```typescript
// Untuk animasi complex dari After Effects
import { Vue3Lottie } from 'vue3-lottie'

<Vue3Lottie
  :animationData="animationJSON"
  :loop="true"
  :autoPlay="true"
/>
```

### GSAP (GreenSock)
**Professional-grade animation**

```bash
npm install gsap
```

```typescript
// Untuk animasi yang sangat complex
// Timeline-based, physics, scroll triggers

import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

// Kapan pakai?
// - Marketing sites
// - Interactive experiences
// - Complex scroll animations
```

---

## Testing

### Vitest
**Unit testing - RECOMMENDED**

```bash
npm install -D vitest @vue/test-utils happy-dom
```

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'happy-dom',
    globals: true,
  },
})
```

### Playwright
**E2E testing - RECOMMENDED**

```bash
npm install -D @playwright/test
```

```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',
  use: {
    baseURL: 'http://localhost:3000',
  },
})

// tests/e2e/login.spec.ts
import { test, expect } from '@playwright/test'

test('user can login', async ({ page }) => {
  await page.goto('/login')
  await page.fill('[name="email"]', 'test@test.com')
  await page.fill('[name="password"]', 'password')
  await page.click('button[type="submit"]')
  await expect(page).toHaveURL('/dashboard')
})
```

### MSW (Mock Service Worker)
**API mocking - RECOMMENDED**

```bash
npm install -D msw
```

```typescript
// Mocking API di tests dan development
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'

const server = setupServer(
  http.get('/api/products', () => {
    return HttpResponse.json([
      { id: '1', name: 'Product 1' },
    ])
  })
)

beforeAll(() => server.listen())
afterAll(() => server.close())
```

---

## DevTools & DX

### Nuxt DevTools
**Built-in Nuxt**

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  devtools: { enabled: true },
})
```

### unplugin-auto-import
**Auto-import utilities**

```bash
npm install -D unplugin-auto-import
```

```typescript
// Nuxt sudah auto-import by default
// Gunakan untuk project non-Nuxt
```

### @nuxt/eslint
**ESLint config untuk Nuxt**

```bash
npm install -D @nuxt/eslint-config eslint
```

```javascript
// eslint.config.mjs
import nuxt from '@nuxt/eslint-config'

export default [
  ...nuxt,
  {
    rules: {
      'vue/multi-word-component-names': 'off',
    },
  },
]
```

---

## Libraries to AVOID

### ❌ Vuex
```
Deprecated. Gunakan Pinia.
```

### ❌ Moment.js
```
Huge bundle size (67KB). Gunakan date-fns atau dayjs.
```

### ❌ Axios (di Nuxt)
```
$fetch sudah cukup dan lebih ringan.
Axios hanya jika butuh fitur specific (interceptors, etc.)
```

### ❌ jQuery
```
Tidak perlu di Vue. Semua bisa dengan Vue + VueUse.
```

### ❌ Bootstrap Vue
```
Maintenance minimal, tidak support Vue 3 properly.
Gunakan Nuxt UI atau PrimeVue.
```

### ❌ Vuetify 2
```
Gunakan Vuetify 3 yang support Vue 3.
Tapi prefer Nuxt UI untuk Nuxt projects.
```

### ❌ vue-resource
```
Deprecated sejak lama. Gunakan $fetch atau TanStack Query.
```

---

## Quick Reference Table

| Category | Recommended | Alternative |
|----------|-------------|-------------|
| Server State | TanStack Query | - |
| UI State | Pinia | - |
| Utilities | VueUse | lodash-es |
| HTTP Client | $fetch | ky, ofetch |
| Form Validation | VeeValidate + Zod | FormKit |
| Schema Validation | Zod | Yup |
| UI Components | Radix Vue | Headless UI, Nuxt UI |
| Date | date-fns | dayjs |
| Animation | @vueuse/motion | GSAP |
| Unit Test | Vitest | Jest |
| E2E Test | Playwright | Cypress |
| ID Generator | nanoid | uuid |
