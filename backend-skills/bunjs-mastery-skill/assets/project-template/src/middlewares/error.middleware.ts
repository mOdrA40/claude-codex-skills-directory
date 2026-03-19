/**
 * Global Error Handler Middleware
 */

import type { MiddlewareHandler } from "hono"
import { AppError, ValidationError } from "../utils/errors"
import { logger } from "../utils/logger"
import { error } from "../utils/response"

export const errorHandler: MiddlewareHandler = async (c, next) => {
  try {
    await next()
  } catch (err) {
    const requestId = c.get("requestId") as string | undefined

    if (err instanceof AppError) {
      // Operational error - expected
      logger.warn(
        {
          err,
          requestId,
          path: c.req.path,
          method: c.req.method,
        },
        err.message
      )

      return c.json(
        error(
          err.code,
          err.message,
          err instanceof ValidationError ? err.errors : undefined
        ),
        err.statusCode
      )
    }

    // Programming error - unexpected
    logger.error(
      {
        err,
        requestId,
        path: c.req.path,
        method: c.req.method,
        stack: err instanceof Error ? err.stack : undefined,
      },
      "Unexpected error"
    )

    return c.json(error("INTERNAL_ERROR", "Something went wrong"), 500)
  }
}
