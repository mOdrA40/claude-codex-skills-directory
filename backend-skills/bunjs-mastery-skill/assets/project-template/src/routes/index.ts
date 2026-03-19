/**
 * Route Aggregator
 * 
 * All routes are registered here.
 */

import { Hono } from "hono"
import { healthRoutes } from "./health.route"
import { userRoutes } from "./user.route"

export const routes = new Hono()

// Health check routes (no prefix)
routes.route("/", healthRoutes)

// API routes
routes.route("/api/users", userRoutes)

// 404 handler
routes.notFound((c) => {
  return c.json(
    {
      success: false,
      error: {
        code: "NOT_FOUND",
        message: `Route ${c.req.method} ${c.req.path} not found`,
      },
    },
    404
  )
})
