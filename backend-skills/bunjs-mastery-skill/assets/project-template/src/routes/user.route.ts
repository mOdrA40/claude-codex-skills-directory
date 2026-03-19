/**
 * User Routes
 * 
 * Example CRUD routes with validation.
 */

import { Hono } from "hono"
import { zValidator } from "@hono/zod-validator"
import { z } from "zod"
import { userService } from "../services/user.service"
import { ok, created, paginated } from "../utils/response"

// Validation schemas
const createUserSchema = z.object({
  email: z.string().email("Invalid email format"),
  name: z.string().min(2, "Name must be at least 2 characters"),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .regex(/[A-Z]/, "Password must contain uppercase letter")
    .regex(/[0-9]/, "Password must contain number"),
})

const updateUserSchema = z.object({
  name: z.string().min(2).optional(),
  email: z.string().email().optional(),
})

const paginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
})

export const userRoutes = new Hono()

// GET /api/users - List users with pagination
userRoutes.get("/", zValidator("query", paginationSchema), async (c) => {
  const { page, limit } = c.req.valid("query")
  const result = await userService.list({ page, limit })

  return c.json(paginated(result.users, { page, limit, total: result.total }))
})

// GET /api/users/:id - Get user by ID
userRoutes.get("/:id", async (c) => {
  const id = c.req.param("id")
  const user = await userService.getById(id)

  return c.json(ok(user))
})

// POST /api/users - Create user
userRoutes.post("/", zValidator("json", createUserSchema), async (c) => {
  const input = c.req.valid("json")
  const user = await userService.create(input)

  return c.json(created(user), 201)
})

// PUT /api/users/:id - Update user
userRoutes.put("/:id", zValidator("json", updateUserSchema), async (c) => {
  const id = c.req.param("id")
  const input = c.req.valid("json")
  const user = await userService.update(id, input)

  return c.json(ok(user))
})

// DELETE /api/users/:id - Delete user
userRoutes.delete("/:id", async (c) => {
  const id = c.req.param("id")
  await userService.delete(id)

  return c.json(ok({ deleted: true }))
})
