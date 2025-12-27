# Folder Structure Guide

## Table of Contents
1. [Feature-Based Structure](#feature-based-structure)
2. [TanStack Start Structure](#tanstack-start-structure)
3. [File Naming Rules](#file-naming-rules)
4. [When to Split Files](#when-to-split-files)

## Feature-Based Structure

Struktur yang PROVEN untuk aplikasi React skala menengah-besar:

```
src/
├── app/                    # App-level setup
│   ├── providers.tsx       # All providers wrapped
│   ├── router.tsx          # Router configuration
│   └── query-client.ts     # Query client setup
│
├── features/               # Feature modules (CORE)
│   ├── auth/
│   │   ├── api/
│   │   │   ├── index.ts           # Export barrel
│   │   │   ├── auth.queries.ts    # Query options factory
│   │   │   └── auth.mutations.ts  # Mutation functions
│   │   ├── components/
│   │   │   ├── LoginForm.tsx
│   │   │   └── AuthGuard.tsx
│   │   ├── hooks/
│   │   │   └── useAuth.ts         # Feature-specific hooks
│   │   ├── types.ts               # Feature types
│   │   └── index.ts               # Public API export
│   │
│   ├── users/
│   │   ├── api/
│   │   │   ├── index.ts
│   │   │   ├── users.queries.ts
│   │   │   └── users.mutations.ts
│   │   ├── components/
│   │   │   ├── UserList.tsx
│   │   │   ├── UserCard.tsx
│   │   │   └── UserForm.tsx
│   │   ├── hooks/
│   │   │   └── useUserFilters.ts
│   │   ├── types.ts
│   │   └── index.ts
│   │
│   └── dashboard/
│       ├── api/
│       ├── components/
│       ├── hooks/
│       └── index.ts
│
├── shared/                 # Shared across features
│   ├── components/
│   │   ├── ui/             # Generic UI (Button, Input, Modal)
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   └── index.ts
│   │   └── layout/         # Layout components
│   │       ├── Header.tsx
│   │       ├── Sidebar.tsx
│   │       └── index.ts
│   ├── hooks/
│   │   ├── useDebounce.ts
│   │   ├── useLocalStorage.ts
│   │   └── index.ts
│   ├── utils/
│   │   ├── format.ts       # formatDate, formatCurrency
│   │   ├── validation.ts   # Zod schemas
│   │   └── index.ts
│   └── types/
│       └── common.ts       # Shared types
│
├── routes/                 # Route components (TanStack Router)
│   ├── __root.tsx          # Root layout
│   ├── index.tsx           # Home route
│   ├── login.tsx
│   ├── dashboard/
│   │   ├── index.tsx
│   │   └── $userId.tsx     # Dynamic route
│   └── users/
│       ├── index.tsx
│       └── $userId.tsx
│
├── lib/                    # External service configs
│   ├── api-client.ts       # Axios/fetch wrapper
│   ├── query-client.ts     # TanStack Query client
│   └── analytics.ts
│
└── config/
    ├── env.ts              # Environment variables (typed)
    └── constants.ts        # App constants
```

## TanStack Start Structure

Untuk full-stack dengan TanStack Start (Vinxi-based):

```
app/
├── routes/                 # File-based routing
│   ├── __root.tsx          # Root layout dengan providers
│   ├── index.tsx           # Home "/"
│   ├── api/                # API routes
│   │   └── users.ts        # /api/users endpoint
│   ├── dashboard/
│   │   ├── index.tsx       # /dashboard
│   │   └── $userId.tsx     # /dashboard/:userId
│   └── auth/
│       ├── login.tsx
│       └── register.tsx
│
├── features/               # Same as above
│   ├── auth/
│   ├── users/
│   └── dashboard/
│
├── shared/                 # Same as above
│
├── server/                 # Server-only code
│   ├── db/
│   │   ├── schema.ts       # Drizzle/Prisma schema
│   │   └── client.ts
│   ├── services/
│   │   └── user.service.ts
│   └── middleware/
│       └── auth.ts
│
└── app.config.ts           # Vinxi/Start config
```

## File Naming Rules

```
Components:     PascalCase.tsx      → UserCard.tsx
Hooks:          camelCase.ts        → useUserProfile.ts
Utils:          kebab-case.ts       → format-date.ts
Types:          kebab-case.ts       → user.types.ts
Queries:        feature.queries.ts  → users.queries.ts
Mutations:      feature.mutations.ts→ users.mutations.ts
Routes:         kebab-case.tsx      → user-profile.tsx
```

## When to Split Files

### Keep Together (Colocation)
```typescript
// ✅ Component + its types di file yang sama
// UserCard.tsx
interface UserCardProps {
  user: User
}

export function UserCard({ user }: UserCardProps) {
  // ...
}
```

### Split When
```typescript
// ✅ Types shared across 3+ files → types.ts
// ✅ More than 5 queries → split by domain
// ✅ Component > 200 lines → extract sub-components
// ✅ Hook used in 3+ places → shared/hooks/
```

## Import Path Aliases

```json
// tsconfig.json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"],
      "@/features/*": ["./src/features/*"],
      "@/shared/*": ["./src/shared/*"]
    }
  }
}
```

Usage:
```typescript
// ✅ Absolute imports
import { UserCard } from '@/features/users'
import { Button } from '@/shared/components/ui'

// ❌ Relative hell
import { UserCard } from '../../../features/users/components/UserCard'
```

## Barrel Exports Pattern

```typescript
// features/users/index.ts - PUBLIC API
export { UserCard } from './components/UserCard'
export { UserList } from './components/UserList'
export { userQueries } from './api/users.queries'
export type { User, UserFilter } from './types'

// ❌ Never export internal implementation
// export { useInternalUserState } from './hooks/useInternalUserState'
```

## Module Boundaries Rule

```
features/auth/ can import from:
  ✅ @/shared/*
  ✅ @/lib/*
  ❌ @/features/users/*  (cross-feature import)
  
If needed, lift to shared/ or create integration layer
```
