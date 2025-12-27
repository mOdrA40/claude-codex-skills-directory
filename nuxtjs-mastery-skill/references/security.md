# Security Best Practices — Vue.js & Nuxt 3

> Security bukan fitur optional. Satu vulnerability bisa menghancurkan bisnis.

## Table of Contents
1. [XSS Prevention](#xss-prevention)
2. [CSRF Protection](#csrf-protection)
3. [Authentication Patterns](#authentication-patterns)
4. [Authorization](#authorization)
5. [Input Validation](#input-validation)
6. [API Security](#api-security)
7. [Security Headers](#security-headers)
8. [Environment Variables](#environment-variables)
9. [Common Vulnerabilities](#common-vulnerabilities)

---

## XSS Prevention

### Vue's Built-in Protection

```vue
<!-- ✅ SAFE: Vue auto-escapes -->
<template>
  <p>{{ userInput }}</p>
  <!-- <script>alert('xss')</script> jadi text biasa -->
</template>

<!-- ❌ DANGEROUS: v-html bypass escaping -->
<template>
  <div v-html="userInput" />
  <!-- JANGAN PERNAH gunakan v-html untuk user input! -->
</template>
```

### Safe v-html Usage

```typescript
// Jika HARUS pakai v-html, sanitize dulu!
import DOMPurify from 'dompurify'

const sanitizedContent = computed(() => {
  return DOMPurify.sanitize(userContent.value, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
    ALLOWED_ATTR: ['href', 'target'],
  })
})

<template>
  <div v-html="sanitizedContent" />
</template>
```

### URL Sanitization

```typescript
// ❌ DANGEROUS: User-controlled URLs
<template>
  <a :href="userProvidedUrl">Click me</a>
  <!-- javascript:alert('xss') bisa lolos! -->
</template>

// ✅ SAFE: Validate URLs
const safeUrl = computed(() => {
  const url = userProvidedUrl.value
  
  // Only allow http/https
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return '#'
  }
  
  // Validate it's a real URL
  try {
    new URL(url)
    return url
  } catch {
    return '#'
  }
})

<template>
  <a :href="safeUrl" rel="noopener noreferrer" target="_blank">
    Click me
  </a>
</template>
```

### Event Handler Injection

```typescript
// ❌ DANGEROUS: Dynamic event handlers
const eventName = ref('onclick') // User controlled!
<button v-on:[eventName]="handler" />

// ✅ SAFE: Whitelist allowed events
const allowedEvents = ['click', 'submit', 'change']
const safeEventName = computed(() => {
  return allowedEvents.includes(eventName.value) ? eventName.value : 'click'
})
```

---

## CSRF Protection

### Nuxt Server CSRF

```typescript
// server/middleware/csrf.ts
import { createHash, randomBytes } from 'crypto'

export default defineEventHandler(async (event) => {
  // Skip untuk GET requests
  if (event.method === 'GET') return

  const cookie = getCookie(event, 'csrf-token')
  const header = getHeader(event, 'x-csrf-token')

  if (!cookie || !header || cookie !== header) {
    throw createError({
      statusCode: 403,
      message: 'Invalid CSRF token',
    })
  }
})

// server/api/csrf.get.ts
export default defineEventHandler((event) => {
  const token = randomBytes(32).toString('hex')
  
  setCookie(event, 'csrf-token', token, {
    httpOnly: true,
    sameSite: 'strict',
    secure: process.env.NODE_ENV === 'production',
    path: '/',
  })
  
  return { token }
})
```

```typescript
// composables/useCsrf.ts
export function useCsrf() {
  const token = ref<string>('')

  async function fetchToken() {
    const { token: newToken } = await $fetch('/api/csrf')
    token.value = newToken
  }

  function getHeaders() {
    return {
      'X-CSRF-Token': token.value,
    }
  }

  return { token, fetchToken, getHeaders }
}

// Usage
const { fetchToken, getHeaders } = useCsrf()
await fetchToken()

await $fetch('/api/products', {
  method: 'POST',
  headers: getHeaders(),
  body: productData,
})
```

---

## Authentication Patterns

### JWT Best Practices

```typescript
// ❌ BAD: JWT di localStorage
localStorage.setItem('token', jwt) // XSS vulnerable!

// ✅ GOOD: HttpOnly cookie
// server/api/auth/login.post.ts
export default defineEventHandler(async (event) => {
  const { email, password } = await readBody(event)
  
  // Validate credentials...
  const user = await validateUser(email, password)
  if (!user) {
    throw createError({ statusCode: 401, message: 'Invalid credentials' })
  }
  
  // Create JWT
  const token = await createJWT({ userId: user.id, role: user.role })
  
  // Set sebagai HttpOnly cookie
  setCookie(event, 'auth-token', token, {
    httpOnly: true,      // Tidak bisa diakses JavaScript
    secure: true,        // HTTPS only
    sameSite: 'strict',  // CSRF protection
    maxAge: 60 * 60 * 24 * 7, // 7 days
    path: '/',
  })
  
  return { user: { id: user.id, name: user.name } }
})
```

### Auth Middleware

```typescript
// server/middleware/auth.ts
export default defineEventHandler(async (event) => {
  // Skip public routes
  const publicPaths = ['/api/auth/login', '/api/auth/register', '/api/public']
  if (publicPaths.some(path => event.path.startsWith(path))) {
    return
  }

  const token = getCookie(event, 'auth-token')
  
  if (!token) {
    throw createError({ statusCode: 401, message: 'Unauthorized' })
  }

  try {
    const payload = await verifyJWT(token)
    event.context.user = payload
  } catch {
    throw createError({ statusCode: 401, message: 'Invalid token' })
  }
})
```

### Session Management

```typescript
// composables/useAuth.ts
export function useAuth() {
  const user = useState<User | null>('auth-user', () => null)
  const isAuthenticated = computed(() => !!user.value)

  async function login(email: string, password: string) {
    const { user: userData } = await $fetch('/api/auth/login', {
      method: 'POST',
      body: { email, password },
    })
    user.value = userData
  }

  async function logout() {
    await $fetch('/api/auth/logout', { method: 'POST' })
    user.value = null
    navigateTo('/login')
  }

  async function refreshUser() {
    try {
      const { user: userData } = await $fetch('/api/auth/me')
      user.value = userData
    } catch {
      user.value = null
    }
  }

  return {
    user: readonly(user),
    isAuthenticated,
    login,
    logout,
    refreshUser,
  }
}
```

---

## Authorization

### Route-level Authorization

```typescript
// middleware/admin.ts
export default defineNuxtRouteMiddleware((to, from) => {
  const { user, isAuthenticated } = useAuth()
  
  if (!isAuthenticated.value) {
    return navigateTo('/login')
  }
  
  if (user.value?.role !== 'admin') {
    return navigateTo('/unauthorized')
  }
})

// pages/admin/dashboard.vue
<script setup>
definePageMeta({
  middleware: ['admin'],
})
</script>
```

### Component-level Authorization

```vue
<!-- components/AdminOnly.vue -->
<script setup lang="ts">
interface Props {
  fallback?: boolean
  requiredRole?: string | string[]
}

const props = withDefaults(defineProps<Props>(), {
  fallback: false,
  requiredRole: 'admin',
})

const { user } = useAuth()

const hasAccess = computed(() => {
  if (!user.value) return false
  
  const requiredRoles = Array.isArray(props.requiredRole) 
    ? props.requiredRole 
    : [props.requiredRole]
  
  return requiredRoles.includes(user.value.role)
})
</script>

<template>
  <slot v-if="hasAccess" />
  <slot v-else-if="fallback" name="fallback" />
</template>

<!-- Usage -->
<AdminOnly>
  <DeleteButton @click="deleteUser" />
  
  <template #fallback>
    <span>You don't have permission</span>
  </template>
</AdminOnly>
```

### API-level Authorization

```typescript
// server/utils/auth.ts
export function requireRole(event: H3Event, roles: string | string[]) {
  const user = event.context.user
  
  if (!user) {
    throw createError({ statusCode: 401, message: 'Unauthorized' })
  }
  
  const allowedRoles = Array.isArray(roles) ? roles : [roles]
  
  if (!allowedRoles.includes(user.role)) {
    throw createError({ statusCode: 403, message: 'Forbidden' })
  }
  
  return user
}

// server/api/admin/users.get.ts
export default defineEventHandler((event) => {
  const user = requireRole(event, 'admin')
  
  // Only admins reach here
  return getUsers()
})
```

---

## Input Validation

### Server-side Validation (WAJIB!)

```typescript
// server/api/products.post.ts
import { z } from 'zod'

const createProductSchema = z.object({
  name: z.string().min(1).max(200),
  price: z.number().positive(),
  description: z.string().max(5000).optional(),
  category: z.enum(['electronics', 'clothing', 'food']),
  sku: z.string().regex(/^[A-Z0-9-]+$/),
})

export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  
  // Validate - throws jika invalid
  const validated = createProductSchema.parse(body)
  
  // Safe to use validated data
  return createProduct(validated)
})
```

### Client-side Validation

```typescript
// composables/useProductForm.ts
import { useForm } from 'vee-validate'
import { toTypedSchema } from '@vee-validate/zod'
import { z } from 'zod'

// SHARE schema antara client dan server!
export const productSchema = z.object({
  name: z.string().min(1, 'Required').max(200),
  price: z.number().positive('Must be positive'),
  category: z.enum(['electronics', 'clothing', 'food']),
})

export function useProductForm() {
  const { handleSubmit, errors, defineField } = useForm({
    validationSchema: toTypedSchema(productSchema),
  })

  const [name, nameAttrs] = defineField('name')
  const [price, priceAttrs] = defineField('price')

  const onSubmit = handleSubmit(async (values) => {
    await $fetch('/api/products', {
      method: 'POST',
      body: values,
    })
  })

  return { name, nameAttrs, price, priceAttrs, errors, onSubmit }
}
```

---

## API Security

### Rate Limiting

```typescript
// server/middleware/rate-limit.ts
const requestCounts = new Map<string, { count: number; resetTime: number }>()

export default defineEventHandler((event) => {
  const ip = getRequestIP(event, { xForwardedFor: true }) || 'unknown'
  const now = Date.now()
  const windowMs = 60 * 1000 // 1 minute
  const maxRequests = 100

  const current = requestCounts.get(ip) || { count: 0, resetTime: now + windowMs }

  if (now > current.resetTime) {
    current.count = 0
    current.resetTime = now + windowMs
  }

  current.count++
  requestCounts.set(ip, current)

  if (current.count > maxRequests) {
    setResponseHeader(event, 'Retry-After', Math.ceil((current.resetTime - now) / 1000))
    throw createError({
      statusCode: 429,
      message: 'Too many requests',
    })
  }
})
```

### API Key Validation

```typescript
// server/middleware/api-key.ts
export default defineEventHandler((event) => {
  // Only untuk API routes
  if (!event.path.startsWith('/api/v1')) return

  const apiKey = getHeader(event, 'x-api-key')
  
  if (!apiKey) {
    throw createError({ statusCode: 401, message: 'API key required' })
  }

  const validKey = await validateApiKey(apiKey)
  if (!validKey) {
    throw createError({ statusCode: 401, message: 'Invalid API key' })
  }

  event.context.apiClient = validKey.client
})
```

---

## Security Headers

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  routeRules: {
    '/**': {
      headers: {
        // Prevent clickjacking
        'X-Frame-Options': 'DENY',
        
        // Prevent MIME sniffing
        'X-Content-Type-Options': 'nosniff',
        
        // Enable XSS filter
        'X-XSS-Protection': '1; mode=block',
        
        // Referrer policy
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        
        // Content Security Policy
        'Content-Security-Policy': [
          "default-src 'self'",
          "script-src 'self' 'unsafe-inline' 'unsafe-eval'", // Nuxt needs these
          "style-src 'self' 'unsafe-inline'",
          "img-src 'self' data: https:",
          "font-src 'self'",
          "connect-src 'self' https://api.example.com",
          "frame-ancestors 'none'",
        ].join('; '),
        
        // HSTS
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        
        // Permissions Policy
        'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
      },
    },
  },
})
```

---

## Environment Variables

### Proper .env Setup

```bash
# .env.example - COMMIT THIS
DATABASE_URL=postgresql://user:pass@localhost:5432/db
JWT_SECRET=your-secret-here
API_KEY=your-api-key

# .env - NEVER COMMIT!
# Add to .gitignore
```

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  runtimeConfig: {
    // Server-only (PRIVATE)
    jwtSecret: process.env.JWT_SECRET,
    databaseUrl: process.env.DATABASE_URL,
    
    // Client-accessible (PUBLIC only!)
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE,
      appName: 'My App',
    },
  },
})

// Usage in server
const config = useRuntimeConfig()
console.log(config.jwtSecret) // Available

// Usage in client
const config = useRuntimeConfig()
console.log(config.public.apiBase) // Available
console.log(config.jwtSecret) // undefined - not exposed!
```

### Validate Environment Variables

```typescript
// server/utils/env.ts
import { z } from 'zod'

const envSchema = z.object({
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  NODE_ENV: z.enum(['development', 'production', 'test']),
})

// Validate on startup
export const env = envSchema.parse(process.env)

// Fail fast if env vars are missing
```

---

## Common Vulnerabilities

### SQL Injection

```typescript
// ❌ DANGEROUS
const query = `SELECT * FROM users WHERE id = ${userId}` // Injection!

// ✅ SAFE: Use parameterized queries
const user = await db.query('SELECT * FROM users WHERE id = $1', [userId])

// ✅ SAFE: Use ORM
const user = await prisma.user.findUnique({ where: { id: userId } })
```

### NoSQL Injection

```typescript
// ❌ DANGEROUS
const user = await db.collection('users').findOne({ name: userInput })
// If userInput = { $gt: '' }, returns all users!

// ✅ SAFE: Validate input type
const user = await db.collection('users').findOne({ 
  name: String(userInput) // Force string
})
```

### Path Traversal

```typescript
// ❌ DANGEROUS
const filePath = `/uploads/${userInput}` // userInput = ../../../etc/passwd

// ✅ SAFE: Validate and sanitize
import path from 'path'

const safePath = path.join('/uploads', path.basename(userInput))
// path.basename removes directory traversal
```

### Open Redirect

```typescript
// ❌ DANGEROUS
const redirectUrl = req.query.next
res.redirect(redirectUrl) // Could redirect to malicious site

// ✅ SAFE: Whitelist allowed redirects
const allowedHosts = ['myapp.com', 'www.myapp.com']
const url = new URL(redirectUrl, 'https://myapp.com')

if (!allowedHosts.includes(url.host)) {
  res.redirect('/')
} else {
  res.redirect(url.toString())
}
```
