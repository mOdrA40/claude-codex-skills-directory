/**
 * Application Setup
 * 
 * Configure middleware and routes here.
 */

import { Hono } from "hono"
import { cors } from "hono/cors"
import { logger as honoLogger } from "hono/logger"
import { secureHeaders } from "hono/secure-headers"
import { timing } from "hono/timing"
import { compress } from "hono/compress"

import { errorHandler } from "./middlewares/error.middleware"
import { requestIdMiddleware } from "./middlewares/request-id.middleware"
import { routes } from "./routes"
import { env } from "./config/env"

export const app = new Hono()

// Global middleware
app.use("*", timing())
app.use("*", requestIdMiddleware)
app.use("*", honoLogger())
app.use("*", secureHeaders())
app.use("*", compress())

// CORS
app.use(
  "*",
  cors({
    origin: env.CORS_ORIGINS?.split(",") || "*",
    allowMethods: ["GET", "POST", "PUT", "DELETE", "PATCH"],
    allowHeaders: ["Content-Type", "Authorization"],
    exposeHeaders: ["X-Request-Id"],
    credentials: true,
  })
)

// Error handler
app.use("*", errorHandler)

// Mount routes
app.route("/", routes)
