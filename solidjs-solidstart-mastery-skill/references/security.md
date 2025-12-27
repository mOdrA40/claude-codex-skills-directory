# Security Best Practices & Authentication Patterns

## Authentication Patterns

### Session-Based Auth with SolidStart

```typescript
// lib/auth/session.ts
import { useSession } from 'vinxi/http';

type SessionData = {
  userId?: string;
  role?: 'user' | 'admin';
  expiresAt?: number;
};

export function getSession() {
  'use server';
  return useSession<SessionData>({
    password: process.env.SESSION_SECRET!,
    cookie: {
      name: 'session',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7, // 7 days
    },
  });
}

export async function createUserSession(userId: string, role: string) {
  'use server';
  const session = await getSession();
  await session.update({
    userId,
    role,
    expiresAt: Date.now() + 7 * 24 * 60 * 60 * 1000,
  });
}

export async function destroySession() {
  'use server';
  const session = await getSession();
  await session.clear();
}

export async function getUser() {
  'use server';
  const session = await getSession();
  const { userId } = session.data;
  
  if (!userId) return null;
  
  return db.user.findUnique({ where: { id: userId } });
}
```

### Protected Routes

```typescript
// middleware.ts
import { createMiddleware } from '@solidjs/start/middleware';
import { getSession } from './lib/auth/session';

export default createMiddleware({
  onRequest: async (event) => {
    const url = new URL(event.request.url);
    const protectedPaths = ['/dashboard', '/settings', '/admin'];
    
    if (protectedPaths.some(p => url.pathname.startsWith(p))) {
      const session = await getSession();
      
      if (!session.data.userId) {
        return new Response(null, {
          status: 302,
          headers: { Location: `/login?redirect=${url.pathname}` },
        });
      }
      
      // Check session expiration
      if (session.data.expiresAt && Date.now() > session.data.expiresAt) {
        await session.clear();
        return new Response(null, {
          status: 302,
          headers: { Location: '/login?expired=true' },
        });
      }
    }
    
    // Admin routes
    if (url.pathname.startsWith('/admin')) {
      const session = await getSession();
      if (session.data.role !== 'admin') {
        return new Response('Forbidden', { status: 403 });
      }
    }
  },
});
```

### JWT Authentication

```typescript
// lib/auth/jwt.ts
import { SignJWT, jwtVerify } from 'jose';

const secret = new TextEncoder().encode(process.env.JWT_SECRET!);
const alg = 'HS256';

export interface JWTPayload {
  sub: string;
  role: string;
  exp: number;
}

export async function createToken(userId: string, role: string): Promise<string> {
  return new SignJWT({ sub: userId, role })
    .setProtectedHeader({ alg })
    .setIssuedAt()
    .setExpirationTime('7d')
    .sign(secret);
}

export async function verifyToken(token: string): Promise<JWTPayload | null> {
  try {
    const { payload } = await jwtVerify(token, secret);
    return payload as JWTPayload;
  } catch {
    return null;
  }
}

export async function refreshToken(token: string): Promise<string | null> {
  const payload = await verifyToken(token);
  if (!payload) return null;
  
  // Only refresh if expiring within 1 day
  const expiresIn = payload.exp * 1000 - Date.now();
  if (expiresIn > 24 * 60 * 60 * 1000) return token;
  
  return createToken(payload.sub, payload.role);
}
```

### OAuth Integration

```typescript
// routes/auth/[provider].ts
import { redirect } from '@solidjs/router';
import type { APIEvent } from '@solidjs/start/server';

const providers = {
  google: {
    authUrl: 'https://accounts.google.com/o/oauth2/v2/auth',
    tokenUrl: 'https://oauth2.googleapis.com/token',
    userInfoUrl: 'https://www.googleapis.com/oauth2/v2/userinfo',
    scopes: ['openid', 'profile', 'email'],
  },
  github: {
    authUrl: 'https://github.com/login/oauth/authorize',
    tokenUrl: 'https://github.com/login/oauth/access_token',
    userInfoUrl: 'https://api.github.com/user',
    scopes: ['read:user', 'user:email'],
  },
};

export async function GET({ params, request }: APIEvent) {
  const provider = params.provider as keyof typeof providers;
  const config = providers[provider];
  
  if (!config) {
    return new Response('Unknown provider', { status: 400 });
  }
  
  const url = new URL(request.url);
  const code = url.searchParams.get('code');
  
  if (!code) {
    // Redirect to OAuth provider
    const authUrl = new URL(config.authUrl);
    authUrl.searchParams.set('client_id', process.env[`${provider.toUpperCase()}_CLIENT_ID`]!);
    authUrl.searchParams.set('redirect_uri', `${url.origin}/auth/${provider}`);
    authUrl.searchParams.set('scope', config.scopes.join(' '));
    authUrl.searchParams.set('response_type', 'code');
    authUrl.searchParams.set('state', crypto.randomUUID());
    
    return redirect(authUrl.toString());
  }
  
  // Exchange code for token
  const tokenResponse = await fetch(config.tokenUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      Accept: 'application/json',
    },
    body: new URLSearchParams({
      client_id: process.env[`${provider.toUpperCase()}_CLIENT_ID`]!,
      client_secret: process.env[`${provider.toUpperCase()}_CLIENT_SECRET`]!,
      code,
      redirect_uri: `${url.origin}/auth/${provider}`,
      grant_type: 'authorization_code',
    }),
  });
  
  const tokens = await tokenResponse.json();
  
  // Get user info
  const userResponse = await fetch(config.userInfoUrl, {
    headers: { Authorization: `Bearer ${tokens.access_token}` },
  });
  
  const profile = await userResponse.json();
  
  // Create or update user
  const user = await upsertUser(provider, profile);
  
  // Create session
  await createUserSession(user.id, user.role);
  
  return redirect('/dashboard');
}
```

## Input Validation & Sanitization

### Zod Validation

```typescript
// lib/validation/schemas.ts
import { z } from 'zod';

// User schemas
export const createUserSchema = z.object({
  email: z.string()
    .email('Invalid email')
    .max(255)
    .transform(v => v.toLowerCase().trim()),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .max(128)
    .regex(/[A-Z]/, 'Must contain uppercase')
    .regex(/[a-z]/, 'Must contain lowercase')
    .regex(/[0-9]/, 'Must contain number'),
  name: z.string()
    .min(2)
    .max(100)
    .transform(v => v.trim()),
});

export const updateUserSchema = createUserSchema.partial();

// Query params
export const paginationSchema = z.object({
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  sort: z.enum(['asc', 'desc']).default('desc'),
  search: z.string().max(100).optional(),
});

// Sanitize HTML content
export const contentSchema = z.string()
  .max(10000)
  .transform(v => sanitizeHtml(v, {
    allowedTags: ['p', 'b', 'i', 'em', 'strong', 'a'],
    allowedAttributes: { a: ['href'] },
  }));
```

### API Route Validation

```typescript
// routes/api/users.ts
import { json } from '@solidjs/router';
import type { APIEvent } from '@solidjs/start/server';
import { createUserSchema, paginationSchema } from '~/lib/validation/schemas';

export async function GET(event: APIEvent) {
  const url = new URL(event.request.url);
  const params = Object.fromEntries(url.searchParams);
  
  const result = paginationSchema.safeParse(params);
  if (!result.success) {
    return json({ error: result.error.flatten() }, { status: 400 });
  }
  
  const { page, limit, sort, search } = result.data;
  
  const users = await db.user.findMany({
    skip: (page - 1) * limit,
    take: limit,
    orderBy: { createdAt: sort },
    where: search ? {
      OR: [
        { name: { contains: search, mode: 'insensitive' } },
        { email: { contains: search, mode: 'insensitive' } },
      ],
    } : undefined,
  });
  
  return json(users);
}

export async function POST(event: APIEvent) {
  const body = await event.request.json();
  
  const result = createUserSchema.safeParse(body);
  if (!result.success) {
    return json({ 
      error: 'Validation failed',
      details: result.error.flatten(),
    }, { status: 400 });
  }
  
  // Check for existing user
  const existing = await db.user.findUnique({
    where: { email: result.data.email },
  });
  
  if (existing) {
    return json({ error: 'Email already registered' }, { status: 409 });
  }
  
  // Hash password
  const hashedPassword = await hashPassword(result.data.password);
  
  const user = await db.user.create({
    data: {
      ...result.data,
      password: hashedPassword,
    },
    select: { id: true, email: true, name: true },
  });
  
  return json(user, { status: 201 });
}
```

## XSS Prevention

### Safe HTML Rendering

```typescript
// ❌ DANGEROUS: Never use innerHTML with user content
function DangerousComponent(props: { html: string }) {
  return <div innerHTML={props.html} />; // XSS vulnerability!
}

// ✅ SAFE: Sanitize before rendering
import DOMPurify from 'dompurify';

function SafeHTML(props: { html: string }) {
  const sanitized = () => DOMPurify.sanitize(props.html, {
    ALLOWED_TAGS: ['p', 'b', 'i', 'em', 'strong', 'a', 'ul', 'li'],
    ALLOWED_ATTR: ['href', 'target'],
  });
  
  return <div innerHTML={sanitized()} />;
}

// ✅ SAFE: Use text content for user input
function UserComment(props: { text: string }) {
  return <p>{props.text}</p>; // SolidJS escapes by default
}
```

### Content Security Policy

```typescript
// middleware.ts
export default createMiddleware({
  onRequest: (event) => {
    const nonce = crypto.randomUUID();
    event.locals.nonce = nonce;
  },
  onBeforeResponse: (event, response) => {
    const nonce = event.locals.nonce;
    
    response.headers.set('Content-Security-Policy', [
      `default-src 'self'`,
      `script-src 'self' 'nonce-${nonce}'`,
      `style-src 'self' 'unsafe-inline'`,
      `img-src 'self' data: https:`,
      `font-src 'self'`,
      `connect-src 'self' https://api.example.com`,
      `frame-ancestors 'none'`,
      `base-uri 'self'`,
      `form-action 'self'`,
    ].join('; '));
    
    response.headers.set('X-Content-Type-Options', 'nosniff');
    response.headers.set('X-Frame-Options', 'DENY');
    response.headers.set('X-XSS-Protection', '1; mode=block');
    response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
    
    return response;
  },
});
```

## CSRF Protection

```typescript
// lib/csrf.ts
import { getSession } from './auth/session';

export async function generateCSRFToken(): Promise<string> {
  'use server';
  const session = await getSession();
  const token = crypto.randomUUID();
  await session.update({ csrfToken: token });
  return token;
}

export async function validateCSRFToken(token: string): Promise<boolean> {
  'use server';
  const session = await getSession();
  return session.data.csrfToken === token;
}

// Component for forms
function CSRFInput() {
  const [token] = createResource(generateCSRFToken);
  
  return <input type="hidden" name="_csrf" value={token() ?? ''} />;
}

// API route validation
export async function POST(event: APIEvent) {
  const formData = await event.request.formData();
  const csrfToken = formData.get('_csrf') as string;
  
  const valid = await validateCSRFToken(csrfToken);
  if (!valid) {
    return json({ error: 'Invalid CSRF token' }, { status: 403 });
  }
  
  // Process request...
}
```

## Rate Limiting

```typescript
// lib/rate-limit.ts
const rateLimit = new Map<string, { count: number; resetAt: number }>();

interface RateLimitConfig {
  windowMs: number;
  max: number;
}

export function checkRateLimit(
  key: string, 
  config: RateLimitConfig = { windowMs: 60000, max: 100 }
): { allowed: boolean; remaining: number; resetAt: number } {
  const now = Date.now();
  const record = rateLimit.get(key);
  
  if (!record || now > record.resetAt) {
    rateLimit.set(key, { count: 1, resetAt: now + config.windowMs });
    return { allowed: true, remaining: config.max - 1, resetAt: now + config.windowMs };
  }
  
  if (record.count >= config.max) {
    return { allowed: false, remaining: 0, resetAt: record.resetAt };
  }
  
  record.count++;
  return { allowed: true, remaining: config.max - record.count, resetAt: record.resetAt };
}

// Usage in API route
export async function POST(event: APIEvent) {
  const ip = event.request.headers.get('x-forwarded-for') || 'unknown';
  const result = checkRateLimit(`login:${ip}`, { windowMs: 60000, max: 5 });
  
  if (!result.allowed) {
    return json(
      { error: 'Too many requests' },
      {
        status: 429,
        headers: {
          'Retry-After': String(Math.ceil((result.resetAt - Date.now()) / 1000)),
          'X-RateLimit-Remaining': '0',
          'X-RateLimit-Reset': String(result.resetAt),
        },
      }
    );
  }
  
  // Process login...
}
```

## SQL Injection Prevention

```typescript
// ✅ Always use parameterized queries

// Prisma (safe by default)
const user = await db.user.findFirst({
  where: { email: userInput }, // Parameterized
});

// Raw SQL with Prisma
const users = await db.$queryRaw`
  SELECT * FROM users WHERE email = ${userInput}
`; // Safe - tagged template

// ❌ NEVER do this
const users = await db.$queryRawUnsafe(
  `SELECT * FROM users WHERE email = '${userInput}'`
); // SQL INJECTION!
```

## Environment Variables

```typescript
// lib/env.ts
import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']),
  DATABASE_URL: z.string().url(),
  SESSION_SECRET: z.string().min(32),
  JWT_SECRET: z.string().min(32),
  GOOGLE_CLIENT_ID: z.string().optional(),
  GOOGLE_CLIENT_SECRET: z.string().optional(),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  console.error('❌ Invalid environment variables:');
  console.error(parsed.error.flatten().fieldErrors);
  process.exit(1);
}

export const env = parsed.data;

// Usage
import { env } from '~/lib/env';
const secret = env.SESSION_SECRET;
```

## Security Headers Checklist

```typescript
// Complete security headers middleware
export default createMiddleware({
  onBeforeResponse: (event, response) => {
    // Prevent clickjacking
    response.headers.set('X-Frame-Options', 'DENY');
    
    // Prevent MIME sniffing
    response.headers.set('X-Content-Type-Options', 'nosniff');
    
    // Enable XSS filter
    response.headers.set('X-XSS-Protection', '1; mode=block');
    
    // Control referrer info
    response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
    
    // Permissions policy
    response.headers.set('Permissions-Policy', 
      'camera=(), microphone=(), geolocation=(), interest-cohort=()'
    );
    
    // HSTS (HTTPS only)
    if (process.env.NODE_ENV === 'production') {
      response.headers.set(
        'Strict-Transport-Security',
        'max-age=31536000; includeSubDomains; preload'
      );
    }
    
    return response;
  },
});
```

## Security Audit Checklist

```markdown
## Pre-deployment Security Checklist

### Authentication
- [ ] Passwords hashed with bcrypt/argon2
- [ ] Session tokens are cryptographically random
- [ ] Sessions expire and can be revoked
- [ ] Login rate limiting implemented
- [ ] Password reset tokens single-use

### Authorization
- [ ] Route guards on protected pages
- [ ] API endpoints verify permissions
- [ ] Role-based access control
- [ ] Resource ownership verified

### Input Validation
- [ ] All inputs validated with Zod
- [ ] File uploads restricted and scanned
- [ ] SQL queries parameterized
- [ ] HTML sanitized before rendering

### Headers & Transport
- [ ] HTTPS enforced in production
- [ ] Security headers configured
- [ ] CORS properly configured
- [ ] Cookies have secure flags

### Secrets
- [ ] Environment variables validated
- [ ] No secrets in source code
- [ ] Keys rotated regularly
- [ ] Secrets not logged
```
