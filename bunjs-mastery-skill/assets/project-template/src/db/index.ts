/**
 * Database Connection
 */

import { drizzle } from "drizzle-orm/postgres-js"
import postgres from "postgres"
import * as schema from "./schema"
import { env } from "../config/env"
import { dbLogger } from "../utils/logger"

// Connection pool configuration
const client = postgres(env.DATABASE_URL, {
  max: 10, // Maximum connections
  idle_timeout: 20, // Close idle connections after 20 seconds
  connect_timeout: 10, // Timeout for new connections
  onnotice: () => {}, // Suppress notices
})

export const db = drizzle(client, {
  schema,
  logger: env.LOG_LEVEL === "debug" ? {
    logQuery: (query, params) => {
      dbLogger.debug({ query, params }, "SQL Query")
    },
  } : undefined,
})

export async function closeDatabase() {
  dbLogger.info("Closing database connections")
  await client.end()
}

export { client }
