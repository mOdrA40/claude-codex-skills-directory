/**
 * Request ID Middleware
 * 
 * Generates unique request ID for tracing.
 */

import type { MiddlewareHandler } from "hono"
import { nanoid } from "nanoid"

export const requestIdMiddleware: MiddlewareHandler = async (c, next) => {
  // Use existing request ID from header or generate new one
  const requestId = c.req.header("x-request-id") || nanoid()

  // Store in context for use in other middleware/handlers
  c.set("requestId", requestId)

  // Add to response header
  c.header("X-Request-Id", requestId)

  await next()
}
