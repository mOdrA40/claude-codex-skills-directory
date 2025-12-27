/**
 * Application Entry Point
 * 
 * This file should be minimal - only server startup logic.
 * All other logic should be in separate modules.
 */

import { app } from "./app"
import { env } from "./config/env"
import { logger } from "./utils/logger"
import { db, closeDatabase } from "./db"

// Start server
const server = Bun.serve({
  port: env.PORT,
  fetch: app.fetch,
})

logger.info({ url: server.url.href }, "🚀 Server started")

// Graceful shutdown
const shutdown = async (signal: string) => {
  logger.info({ signal }, "Shutdown signal received")

  // Stop accepting new connections
  server.stop()

  // Close database connections
  await closeDatabase()

  logger.info("Graceful shutdown complete")
  process.exit(0)
}

process.on("SIGTERM", () => shutdown("SIGTERM"))
process.on("SIGINT", () => shutdown("SIGINT"))

// Unhandled errors
process.on("uncaughtException", (err) => {
  logger.fatal({ err }, "Uncaught Exception")
  process.exit(1)
})

process.on("unhandledRejection", (reason) => {
  logger.error({ reason }, "Unhandled Rejection")
})
