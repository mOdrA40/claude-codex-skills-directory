# Clean Code Patterns untuk Bun.js

## Table of Contents
1. [Naming Conventions](#naming-conventions)
2. [Function Design](#function-design)
3. [Error Handling Patterns](#error-handling-patterns)
4. [Type Safety](#type-safety)
5. [Anti-Patterns to Avoid](#anti-patterns)
6. [Refactoring Examples](#refactoring-examples)

---

## Naming Conventions

### Variables & Functions
```typescript
// ❌ Bad
const d = new Date()
const u = await getU(1)
function proc(x: any) { ... }

// ✅ Good
const createdAt = new Date()
const user = await getUserById(1)
function processPayment(transaction: Transaction) { ... }
```

### Boolean Naming
```typescript
// ❌ Bad
const open = true
const disable = false
const notEmpty = arr.length > 0

// ✅ Good - selalu prefix dengan is/has/can/should
const isOpen = true
const isDisabled = false  // hindari negasi ganda
const hasItems = arr.length > 0
const canEdit = user.role === "admin"
const shouldRefetch = isStale && !isLoading
```

### Constants
```typescript
// ❌ Bad
const timeout = 5000
const max = 100

// ✅ Good - SCREAMING_SNAKE_CASE untuk constants
const REQUEST_TIMEOUT_MS = 5000
const MAX_RETRY_ATTEMPTS = 100
const DEFAULT_PAGE_SIZE = 20
```

### Files & Folders
```
❌ Bad                    ✅ Good
userService.ts           user.service.ts
UserController.ts        user.controller.ts
helperFunctions.ts       string.util.ts
misc.ts                  (hapus - terlalu generic)
```

---

## Function Design

### Single Responsibility
```typescript
// ❌ Bad - melakukan terlalu banyak hal
async function handleUserRegistration(data: UserInput) {
  // validate
  if (!data.email) throw new Error("Email required")
  if (!data.password) throw new Error("Password required")
  
  // hash password
  const hashed = await Bun.password.hash(data.password)
  
  // save to db
  const user = await db.insert(users).values({ ...data, password: hashed })
  
  // send email
  await sendWelcomeEmail(user.email)
  
  // create audit log
  await createAuditLog("USER_CREATED", user.id)
  
  return user
}

// ✅ Good - setiap function punya 1 tanggung jawab
async function registerUser(input: UserInput): Promise<User> {
  const validated = validateUserInput(input)
  const user = await createUser(validated)
  await onUserCreated(user)  // event-driven untuk side effects
  return user
}

function validateUserInput(input: UserInput): ValidatedUserInput {
  const schema = z.object({
    email: z.string().email(),
    password: z.string().min(8),
  })
  return schema.parse(input)
}

async function createUser(input: ValidatedUserInput): Promise<User> {
  const hashedPassword = await hashPassword(input.password)
  return db.insert(users).values({ ...input, password: hashedPassword })
}

async function onUserCreated(user: User): Promise<void> {
  await Promise.all([
    sendWelcomeEmail(user.email),
    createAuditLog("USER_CREATED", user.id),
  ])
}
```

### Early Return Pattern
```typescript
// ❌ Bad - deeply nested
function processOrder(order: Order) {
  if (order) {
    if (order.items.length > 0) {
      if (order.status === "pending") {
        if (order.total > 0) {
          // actual logic here
          return calculateDiscount(order)
        }
      }
    }
  }
  return null
}

// ✅ Good - guard clauses with early return
function processOrder(order: Order | null): Discount | null {
  if (!order) return null
  if (order.items.length === 0) return null
  if (order.status !== "pending") return null
  if (order.total <= 0) return null
  
  return calculateDiscount(order)
}
```

### Pure Functions Preferred
```typescript
// ❌ Impure - mutates input
function addTax(prices: number[]) {
  for (let i = 0; i < prices.length; i++) {
    prices[i] = prices[i] * 1.1
  }
  return prices
}

// ✅ Pure - returns new array
function addTax(prices: number[], taxRate = 0.1): number[] {
  return prices.map(price => price * (1 + taxRate))
}
```

---

## Error Handling Patterns

### Custom Error Classes
```typescript
// src/utils/errors.ts
export abstract class AppError extends Error {
  abstract readonly statusCode: number
  abstract readonly code: string
  readonly isOperational = true
  
  constructor(message: string) {
    super(message)
    Object.setPrototypeOf(this, new.target.prototype)
    Error.captureStackTrace(this, this.constructor)
  }
}

export class NotFoundError extends AppError {
  readonly statusCode = 404
  readonly code = "NOT_FOUND"
  
  constructor(resource: string, id?: string) {
    super(id ? `${resource} with id ${id} not found` : `${resource} not found`)
  }
}

export class UnauthorizedError extends AppError {
  readonly statusCode = 401
  readonly code = "UNAUTHORIZED"
  
  constructor(message = "Authentication required") {
    super(message)
  }
}

export class ForbiddenError extends AppError {
  readonly statusCode = 403
  readonly code = "FORBIDDEN"
  
  constructor(message = "Access denied") {
    super(message)
  }
}

export class ConflictError extends AppError {
  readonly statusCode = 409
  readonly code = "CONFLICT"
  
  constructor(message: string) {
    super(message)
  }
}

export class ValidationError extends AppError {
  readonly statusCode = 400
  readonly code = "VALIDATION_ERROR"
  readonly errors: Record<string, string[]>
  
  constructor(errors: Record<string, string[]>) {
    super("Validation failed")
    this.errors = errors
  }
}
```

### Result Pattern (Alternative to Try-Catch)
```typescript
// Untuk operasi yang EXPECTED bisa gagal
type Result<T, E = Error> = 
  | { ok: true; value: T }
  | { ok: false; error: E }

function parseJson<T>(json: string): Result<T, SyntaxError> {
  try {
    return { ok: true, value: JSON.parse(json) }
  } catch (e) {
    return { ok: false, error: e as SyntaxError }
  }
}

// Usage
const result = parseJson<User>(rawJson)
if (result.ok) {
  console.log(result.value.name)
} else {
  console.error("Invalid JSON:", result.error.message)
}
```

### Global Error Handler (Hono)
```typescript
// src/middlewares/error.middleware.ts
import { Context, MiddlewareHandler } from "hono"
import { AppError } from "../utils/errors"
import { logger } from "../utils/logger"

export const errorHandler: MiddlewareHandler = async (c, next) => {
  try {
    await next()
  } catch (err) {
    if (err instanceof AppError) {
      // Operational error - expected, log as warning
      logger.warn({ err, path: c.req.path }, err.message)
      return c.json({
        success: false,
        error: {
          code: err.code,
          message: err.message,
          ...(err instanceof ValidationError && { details: err.errors }),
        },
      }, err.statusCode)
    }
    
    // Programming error - unexpected, log as error
    logger.error({ err, path: c.req.path }, "Unexpected error")
    return c.json({
      success: false,
      error: {
        code: "INTERNAL_ERROR",
        message: "Something went wrong",
      },
    }, 500)
  }
}
```

---

## Type Safety

### Avoid `any` - Use `unknown` When Needed
```typescript
// ❌ Bad
function process(data: any) {
  return data.value  // no type checking
}

// ✅ Good
function process(data: unknown): string {
  if (isValidData(data)) {
    return data.value
  }
  throw new ValidationError("Invalid data")
}

function isValidData(data: unknown): data is { value: string } {
  return (
    typeof data === "object" &&
    data !== null &&
    "value" in data &&
    typeof (data as { value: unknown }).value === "string"
  )
}
```

### Discriminated Unions
```typescript
// ✅ Type-safe state handling
type AsyncState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: Error }

function handleState<T>(state: AsyncState<T>) {
  switch (state.status) {
    case "idle":
      return "Ready to start"
    case "loading":
      return "Loading..."
    case "success":
      return `Got: ${state.data}`  // TypeScript knows data exists
    case "error":
      return `Error: ${state.error.message}`  // TypeScript knows error exists
  }
}
```

### Branded Types untuk Type Safety Ekstra
```typescript
// Prevent mixing up IDs
type UserId = string & { readonly __brand: "UserId" }
type OrderId = string & { readonly __brand: "OrderId" }

function createUserId(id: string): UserId {
  return id as UserId
}

function getUser(id: UserId) { ... }
function getOrder(id: OrderId) { ... }

const userId = createUserId("user_123")
const orderId = "order_456" as OrderId

getUser(userId)   // ✅ OK
getUser(orderId)  // ❌ Type error! Tidak bisa mix
```

---

## Anti-Patterns

### 1. God Object/File
```typescript
// ❌ Bad - utils.ts dengan 1000+ lines
export function formatDate() { ... }
export function validateEmail() { ... }
export function calculateTax() { ... }
export function sendEmail() { ... }
export function hashPassword() { ... }
// ... 50 more functions

// ✅ Good - split by domain
// date.util.ts
export function formatDate() { ... }

// validation.util.ts  
export function validateEmail() { ... }

// tax.util.ts
export function calculateTax() { ... }
```

### 2. Magic Numbers/Strings
```typescript
// ❌ Bad
if (user.role === "admin") { ... }
if (retries > 3) { ... }
setTimeout(fn, 86400000)

// ✅ Good
const ROLES = { ADMIN: "admin", USER: "user" } as const
const MAX_RETRIES = 3
const ONE_DAY_MS = 24 * 60 * 60 * 1000

if (user.role === ROLES.ADMIN) { ... }
if (retries > MAX_RETRIES) { ... }
setTimeout(fn, ONE_DAY_MS)
```

### 3. Callback Hell
```typescript
// ❌ Bad
getUser(id, (user) => {
  getOrders(user.id, (orders) => {
    getProducts(orders[0].id, (products) => {
      // nightmare
    })
  })
})

// ✅ Good
const user = await getUser(id)
const orders = await getOrders(user.id)
const products = await getProducts(orders[0].id)

// Atau parallel jika tidak dependent
const [user, settings, notifications] = await Promise.all([
  getUser(id),
  getSettings(id),
  getNotifications(id),
])
```

---

## Refactoring Examples

### Before & After: User Service
```typescript
// ❌ Before - monolithic, hard to test
class UserService {
  async register(email: string, password: string, name: string) {
    // validation mixed with business logic
    if (!email.includes("@")) throw new Error("Invalid email")
    if (password.length < 8) throw new Error("Password too short")
    
    // business logic
    const existing = await db.query.users.findFirst({ 
      where: eq(users.email, email) 
    })
    if (existing) throw new Error("Email exists")
    
    const hashed = await Bun.password.hash(password)
    const user = await db.insert(users).values({ 
      email, password: hashed, name 
    }).returning()
    
    // side effects
    await this.sendWelcomeEmail(user[0].email)
    await this.createAuditLog("user_created", user[0].id)
    
    return user[0]
  }
}

// ✅ After - separated concerns, testable
// user.schema.ts
export const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  name: z.string().min(2),
})

export type RegisterInput = z.infer<typeof registerSchema>

// user.repository.ts
export const userRepository = {
  findByEmail: (email: string) =>
    db.query.users.findFirst({ where: eq(users.email, email) }),
  
  create: (data: NewUser) =>
    db.insert(users).values(data).returning().then(r => r[0]),
}

// user.service.ts
export const userService = {
  async register(input: RegisterInput): Promise<User> {
    const existing = await userRepository.findByEmail(input.email)
    if (existing) {
      throw new ConflictError("Email already registered")
    }
    
    const hashedPassword = await Bun.password.hash(input.password)
    const user = await userRepository.create({
      ...input,
      password: hashedPassword,
    })
    
    // Event-driven side effects
    eventBus.emit("user.created", user)
    
    return user
  },
}

// user.controller.ts
export const userController = new Hono()
  .post("/register", zValidator("json", registerSchema), async (c) => {
    const input = c.req.valid("json")
    const user = await userService.register(input)
    return c.json(created(user), 201)
  })
```
