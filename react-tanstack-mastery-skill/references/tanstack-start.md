# TanStack Start Guide

## Table of Contents
1. [Overview](#overview)
2. [Project Setup](#project-setup)
3. [Server Functions](#server-functions)
4. [API Routes](#api-routes)
5. [Data Loading](#data-loading)
6. [Authentication](#authentication)
7. [Middleware](#middleware)
8. [Deployment](#deployment)

## Overview

TanStack Start adalah full-stack React framework berbasis Vinxi (Nitro + Vite). Fitur utama:
- File-based routing (TanStack Router)
- Server functions (RPC-style)
- SSR/SSG/ISR support
- Full TypeScript support
- Edge-ready

## Project Setup

```bash
# Create new project
pnpm create @tanstack/start my-app
cd my-app
pnpm install
pnpm dev
```

### Project Structure
```
my-app/
├── app/
│   ├── routes/
│   │   ├── __root.tsx
│   │   ├── index.tsx
│   │   └── api/
│   │       └── [...].ts
│   ├── client.tsx         # Client entry
│   ├── router.tsx         # Router config
│   └── ssr.tsx            # SSR entry
├── app.config.ts          # Vinxi config
├── package.json
└── tsconfig.json
```

### Configuration
```typescript
// app.config.ts
import { defineConfig } from '@tanstack/start/config'
import tsConfigPaths from 'vite-tsconfig-paths'

export default defineConfig({
  vite: {
    plugins: [tsConfigPaths()],
  },
  server: {
    preset: 'vercel', // atau 'node', 'cloudflare', dll
  },
})
```

## Server Functions

**Server functions = RPC calls yang type-safe**

```typescript
// app/server/functions/users.ts
import { createServerFn } from '@tanstack/start'
import { z } from 'zod'
import { db } from '../db'

// GET users
export const getUsers = createServerFn('GET', async () => {
  const users = await db.user.findMany()
  return users
})

// GET user by ID
export const getUser = createServerFn('GET', async (id: string) => {
  const user = await db.user.findUnique({ where: { id } })
  if (!user) {
    throw new Error('User not found')
  }
  return user
})

// POST create user dengan validation
const createUserSchema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
})

export const createUser = createServerFn(
  'POST',
  async (input: z.infer<typeof createUserSchema>) => {
    // Validate
    const data = createUserSchema.parse(input)
    
    // Create
    const user = await db.user.create({ data })
    return user
  }
)

// POST update user
export const updateUser = createServerFn(
  'POST',
  async ({ id, data }: { id: string; data: Partial<UserInput> }) => {
    const user = await db.user.update({
      where: { id },
      data,
    })
    return user
  }
)

// DELETE user
export const deleteUser = createServerFn('POST', async (id: string) => {
  await db.user.delete({ where: { id } })
  return { success: true }
})
```

### Using Server Functions in Components

```typescript
// app/routes/users/index.tsx
import { createFileRoute, useRouter } from '@tanstack/react-router'
import { useSuspenseQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getUsers, createUser, deleteUser } from '@/server/functions/users'

export const Route = createFileRoute('/users')({
  loader: async () => {
    // Prefetch di server
    return getUsers()
  },
  component: UsersPage,
})

function UsersPage() {
  const queryClient = useQueryClient()
  
  // Use dengan TanStack Query
  const { data: users } = useSuspenseQuery({
    queryKey: ['users'],
    queryFn: () => getUsers(),
  })

  const createMutation = useMutation({
    mutationFn: createUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
    },
  })

  return (
    <div>
      <button onClick={() => createMutation.mutate({ name: 'New', email: 'new@test.com' })}>
        Add User
      </button>
      
      <ul>
        {users.map((user) => (
          <li key={user.id}>
            {user.name}
            <button onClick={() => deleteMutation.mutate(user.id)}>
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
```

## API Routes

```typescript
// app/routes/api/users.ts
import { json } from '@tanstack/start'
import { createAPIFileRoute } from '@tanstack/start/api'
import { db } from '@/server/db'

export const Route = createAPIFileRoute('/api/users')({
  // GET /api/users
  GET: async ({ request }) => {
    const url = new URL(request.url)
    const status = url.searchParams.get('status')
    
    const users = await db.user.findMany({
      where: status ? { status } : undefined,
    })
    
    return json(users)
  },

  // POST /api/users
  POST: async ({ request }) => {
    const body = await request.json()
    
    const user = await db.user.create({ data: body })
    
    return json(user, { status: 201 })
  },
})

// app/routes/api/users/$id.ts
export const Route = createAPIFileRoute('/api/users/$id')({
  // GET /api/users/:id
  GET: async ({ params }) => {
    const user = await db.user.findUnique({
      where: { id: params.id },
    })
    
    if (!user) {
      return json({ error: 'Not found' }, { status: 404 })
    }
    
    return json(user)
  },

  // PATCH /api/users/:id
  PATCH: async ({ params, request }) => {
    const body = await request.json()
    
    const user = await db.user.update({
      where: { id: params.id },
      data: body,
    })
    
    return json(user)
  },

  // DELETE /api/users/:id
  DELETE: async ({ params }) => {
    await db.user.delete({ where: { id: params.id } })
    return json({ success: true })
  },
})
```

## Data Loading

```typescript
// Route loader dengan SSR
export const Route = createFileRoute('/dashboard')({
  // Validate search params
  validateSearch: (search) => dashboardSearchSchema.parse(search),
  
  // Dependencies untuk loader
  loaderDeps: ({ search }) => ({ search }),
  
  // Server-side loader
  loader: async ({ context, deps }) => {
    const [stats, recentActivity] = await Promise.all([
      getStats(),
      getRecentActivity(deps.search),
    ])
    
    return { stats, recentActivity }
  },
  
  // Meta tags untuk SEO
  meta: ({ loaderData }) => [
    { title: `Dashboard - ${loaderData.stats.totalUsers} users` },
    { name: 'description', content: 'Admin dashboard' },
  ],
  
  component: DashboardPage,
})

function DashboardPage() {
  // Loader data sudah tersedia (no loading state needed)
  const { stats, recentActivity } = Route.useLoaderData()
  
  return (
    <div>
      <StatsCards stats={stats} />
      <ActivityFeed items={recentActivity} />
    </div>
  )
}
```

## Authentication

```typescript
// app/server/auth.ts
import { createServerFn } from '@tanstack/start'
import { getCookie, setCookie, deleteCookie } from 'vinxi/http'
import { db } from './db'

export const getSession = createServerFn('GET', async () => {
  const sessionId = getCookie('session')
  if (!sessionId) return null
  
  const session = await db.session.findUnique({
    where: { id: sessionId },
    include: { user: true },
  })
  
  return session
})

export const login = createServerFn(
  'POST',
  async ({ email, password }: { email: string; password: string }) => {
    const user = await db.user.findUnique({ where: { email } })
    
    if (!user || !verifyPassword(password, user.passwordHash)) {
      throw new Error('Invalid credentials')
    }
    
    const session = await db.session.create({
      data: { userId: user.id },
    })
    
    setCookie('session', session.id, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7, // 7 days
    })
    
    return { user }
  }
)

export const logout = createServerFn('POST', async () => {
  const sessionId = getCookie('session')
  if (sessionId) {
    await db.session.delete({ where: { id: sessionId } })
    deleteCookie('session')
  }
  return { success: true }
})

// app/routes/__root.tsx
export const Route = createRootRouteWithContext<RouterContext>()({
  beforeLoad: async () => {
    const session = await getSession()
    return { auth: session }
  },
  component: RootLayout,
})

// Protected route
export const Route = createFileRoute('/_authenticated')({
  beforeLoad: async ({ context, location }) => {
    if (!context.auth) {
      throw redirect({
        to: '/login',
        search: { redirect: location.href },
      })
    }
  },
})
```

## Middleware

```typescript
// app/server/middleware.ts
import { createMiddleware } from '@tanstack/start'
import { getSession } from './auth'

// Auth middleware
export const authMiddleware = createMiddleware().server(async ({ next }) => {
  const session = await getSession()
  
  return next({
    context: {
      user: session?.user ?? null,
    },
  })
})

// Rate limiting middleware
export const rateLimitMiddleware = createMiddleware().server(
  async ({ next, context }) => {
    const ip = context.request.headers.get('x-forwarded-for')
    
    const isAllowed = await checkRateLimit(ip)
    if (!isAllowed) {
      throw new Error('Too many requests')
    }
    
    return next()
  }
)

// Compose middleware
export const protectedServerFn = createServerFn('POST', async () => {
  // ...
}).middleware([authMiddleware, rateLimitMiddleware])
```

## Deployment

### Vercel
```typescript
// app.config.ts
export default defineConfig({
  server: {
    preset: 'vercel',
  },
})
```

### Node.js
```typescript
// app.config.ts
export default defineConfig({
  server: {
    preset: 'node-server',
  },
})
```

```bash
pnpm build
node .output/server/index.mjs
```

### Cloudflare Workers
```typescript
// app.config.ts
export default defineConfig({
  server: {
    preset: 'cloudflare-pages',
  },
})
```

### Docker
```dockerfile
FROM node:20-slim AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install
COPY . .
RUN pnpm build

FROM node:20-slim
WORKDIR /app
COPY --from=builder /app/.output .output
EXPOSE 3000
CMD ["node", ".output/server/index.mjs"]
```

## Environment Variables

```typescript
// app/server/env.ts
import { z } from 'zod'

const envSchema = z.object({
  DATABASE_URL: z.string().url(),
  SESSION_SECRET: z.string().min(32),
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
})

export const env = envSchema.parse(process.env)

// Usage in server functions
import { env } from './env'

export const getConfig = createServerFn('GET', async () => {
  return {
    isProduction: env.NODE_ENV === 'production',
  }
})
```
