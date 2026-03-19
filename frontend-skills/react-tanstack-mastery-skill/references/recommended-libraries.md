# Recommended Libraries

## Table of Contents
1. [Core Stack](#core-stack)
2. [State Management](#state-management)
3. [Forms & Validation](#forms--validation)
4. [UI Components](#ui-components)
5. [Utilities](#utilities)
6. [Testing](#testing)
7. [Development Tools](#development-tools)
8. [Library Selection Criteria](#library-selection-criteria)

## Core Stack

### Minimal Production Setup
```bash
# Foundation
pnpm add react react-dom
pnpm add @tanstack/react-query @tanstack/react-router

# TypeScript
pnpm add -D typescript @types/react @types/react-dom

# Build tool
pnpm add -D vite @vitejs/plugin-react
```

### Full-Stack Setup (TanStack Start)
```bash
pnpm create @tanstack/start my-app
cd my-app
pnpm add @tanstack/react-query zod drizzle-orm
```

## State Management

### Server State: TanStack Query (WAJIB)
```bash
pnpm add @tanstack/react-query
pnpm add -D @tanstack/react-query-devtools
```

### Client State: Zustand (bukan Redux)
```bash
pnpm add zustand
```

```typescript
// stores/ui.store.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface UIStore {
  sidebarOpen: boolean
  theme: 'light' | 'dark'
  toggleSidebar: () => void
  setTheme: (theme: 'light' | 'dark') => void
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      theme: 'light',
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setTheme: (theme) => set({ theme }),
    }),
    { name: 'ui-storage' }
  )
)
```

**Kenapa Zustand > Redux?**
- 90% less boilerplate
- No providers needed
- Built-in persistence
- Simpler mental model
- 2KB vs 40KB+

### URL State: TanStack Router Search Params
```typescript
// Lihat tanstack-router.md untuk detail
```

## Forms & Validation

### Option A: TanStack Form (untuk forms complex)
```bash
pnpm add @tanstack/react-form @tanstack/zod-form-adapter zod
```

### Option B: React Hook Form (mature, banyak integrations)
```bash
pnpm add react-hook-form @hookform/resolvers zod
```

```typescript
// With Zod validation
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'

const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
})

function LoginForm() {
  const { register, handleSubmit, formState: { errors } } = useForm({
    resolver: zodResolver(schema),
  })

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('email')} />
      {errors.email && <span>{errors.email.message}</span>}
    </form>
  )
}
```

### Validation: Zod (WAJIB)
```bash
pnpm add zod
```

```typescript
// shared/utils/validation.ts
import { z } from 'zod'

export const emailSchema = z.string().email('Invalid email')

export const passwordSchema = z.string()
  .min(8, 'Min 8 characters')
  .regex(/[A-Z]/, 'Need uppercase')
  .regex(/[0-9]/, 'Need number')

export const userSchema = z.object({
  name: z.string().min(2).max(100),
  email: emailSchema,
  age: z.number().min(0).max(150).optional(),
})

export type User = z.infer<typeof userSchema>
```

## UI Components

### Option A: Shadcn/ui (RECOMMENDED)
```bash
pnpm dlx shadcn-ui@latest init
pnpm dlx shadcn-ui@latest add button input card
```

**Kenapa Shadcn?**
- Copy-paste, bukan dependency
- Full control & customization
- Radix primitives (accessible)
- Tailwind styling

### Option B: Radix Primitives (headless)
```bash
pnpm add @radix-ui/react-dialog @radix-ui/react-dropdown-menu
```

### Styling: Tailwind CSS
```bash
pnpm add -D tailwindcss postcss autoprefixer
pnpm add tailwind-merge clsx
```

```typescript
// lib/utils.ts
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Usage
<button className={cn(
  'px-4 py-2 rounded',
  isActive && 'bg-blue-500',
  className
)} />
```

### Icons: Lucide React
```bash
pnpm add lucide-react
```

```typescript
import { Search, User, Settings } from 'lucide-react'

<Search className="w-4 h-4" />
```

## Utilities

### Date Handling: date-fns
```bash
pnpm add date-fns
```

```typescript
import { format, formatDistanceToNow, isAfter } from 'date-fns'
import { id } from 'date-fns/locale'

format(new Date(), 'dd MMMM yyyy', { locale: id }) // "27 Desember 2025"
formatDistanceToNow(date, { addSuffix: true }) // "2 hours ago"
```

### HTTP Client: ky atau axios
```bash
# Lightweight (6KB)
pnpm add ky

# Full-featured (30KB)
pnpm add axios
```

```typescript
// lib/api-client.ts (ky)
import ky from 'ky'

export const api = ky.create({
  prefixUrl: '/api',
  timeout: 30000,
  hooks: {
    beforeRequest: [
      (request) => {
        const token = getToken()
        if (token) {
          request.headers.set('Authorization', `Bearer ${token}`)
        }
      },
    ],
  },
})

// Usage
const users = await api.get('users').json<User[]>()
```

### Environment Variables: @t3-oss/env-core
```bash
pnpm add @t3-oss/env-core zod
```

```typescript
// env.ts
import { createEnv } from '@t3-oss/env-core'
import { z } from 'zod'

export const env = createEnv({
  server: {
    DATABASE_URL: z.string().url(),
    JWT_SECRET: z.string().min(32),
  },
  client: {
    NEXT_PUBLIC_API_URL: z.string().url(),
  },
  runtimeEnv: process.env,
})

// Type-safe usage
console.log(env.DATABASE_URL) // Typed & validated
```

## Testing

### Unit Testing: Vitest
```bash
pnpm add -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['./test/setup.ts'],
    globals: true,
  },
})

// test/setup.ts
import '@testing-library/jest-dom'
```

### E2E Testing: Playwright
```bash
pnpm add -D @playwright/test
pnpm exec playwright install
```

```typescript
// e2e/login.spec.ts
import { test, expect } from '@playwright/test'

test('user can login', async ({ page }) => {
  await page.goto('/login')
  await page.fill('[name=email]', 'test@example.com')
  await page.fill('[name=password]', 'password123')
  await page.click('button[type=submit]')
  await expect(page).toHaveURL('/dashboard')
})
```

### API Mocking: MSW
```bash
pnpm add -D msw
```

```typescript
// mocks/handlers.ts
import { http, HttpResponse } from 'msw'

export const handlers = [
  http.get('/api/users', () => {
    return HttpResponse.json([
      { id: '1', name: 'John' },
    ])
  }),
]

// mocks/server.ts (for tests)
import { setupServer } from 'msw/node'
import { handlers } from './handlers'

export const server = setupServer(...handlers)
```

## Development Tools

### Code Quality
```bash
# Linting
pnpm add -D eslint @typescript-eslint/eslint-plugin @typescript-eslint/parser
pnpm add -D eslint-plugin-react-hooks

# Formatting
pnpm add -D prettier

# Git hooks
pnpm add -D husky lint-staged
```

### Bundle Analysis
```bash
pnpm add -D vite-bundle-visualizer
```

```typescript
// vite.config.ts
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig({
  plugins: [
    visualizer({ open: true }),
  ],
})
```

## Library Selection Criteria

### Checklist Sebelum Adopt Library

```markdown
□ Bundle size acceptable? (bundlephobia.com)
□ TypeScript support first-class?
□ Active maintenance? (last commit < 3 months)
□ Adequate downloads? (> 10k/week untuk production)
□ Good documentation?
□ Escape hatch available? (tidak lock-in)
□ Team familiar atau learning curve acceptable?
```

### Red Flags - AVOID
- Last commit > 1 year ago
- No TypeScript types
- < 1k weekly downloads
- No documentation
- Breaking changes setiap minor version
- Abandoned by original maintainer

### Library Comparison Cheatsheet

| Need | ✅ Recommended | ❌ Avoid |
|------|---------------|---------|
| Server state | TanStack Query | SWR (less features) |
| Client state | Zustand | Redux (overkill) |
| Routing | TanStack Router | React Router v5 |
| Forms | RHF / TanStack Form | Formik (outdated) |
| Validation | Zod | Yup (worse TS) |
| UI Components | Shadcn/ui | Material UI (heavy) |
| Styling | Tailwind | Styled-components |
| Date | date-fns | Moment.js (deprecated) |
| HTTP | ky / axios | fetch wrapper sendiri |
| Testing | Vitest | Jest (slower) |
| E2E | Playwright | Cypress (slower) |

### Version Pinning Strategy

```json
// package.json
{
  "dependencies": {
    // Pin exact versions untuk stability
    "@tanstack/react-query": "5.62.0",
    "react": "18.3.1",
    
    // Allow patches untuk utilities
    "date-fns": "^3.6.0",
    "zod": "^3.23.0"
  }
}
```
