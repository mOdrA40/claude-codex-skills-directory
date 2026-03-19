/**
 * Structured Logger
 * 
 * Uses Pino for high-performance JSON logging.
 */

import pino from "pino"
import { env, isDev } from "../config/env"

export const logger = pino({
  level: env.LOG_LEVEL,

  // Pretty print untuk development
  transport: isDev
    ? {
        target: "pino-pretty",
        options: {
          colorize: true,
          translateTime: "SYS:standard",
          ignore: "pid,hostname",
        },
      }
    : undefined,

  // Format level sebagai string
  formatters: {
    level: (label) => ({ level: label }),
  },

  // Redact sensitive data
  redact: {
    paths: ["password", "token", "authorization", "cookie", "*.password", "*.token"],
    censor: "[REDACTED]",
  },

  // Base fields untuk semua logs
  base: {
    env: env.NODE_ENV,
    version: process.env.npm_package_version,
  },
})

// Child loggers untuk modules
export const createLogger = (module: string) => logger.child({ module })

// Pre-configured loggers
export const dbLogger = createLogger("database")
export const authLogger = createLogger("auth")
export const apiLogger = createLogger("api")
