/**
 * Health Check Routes
 */

import { Hono } from "hono"
import { db } from "../db"
import { sql } from "drizzle-orm"

export const healthRoutes = new Hono()

const startTime = Date.now()

// Simple health check (untuk load balancer)
healthRoutes.get("/health", (c) => {
  return c.json({ status: "ok" })
})

// Detailed health check
healthRoutes.get("/health/detailed", async (c) => {
  const dbCheck = await checkDatabase()
  const memory = getMemoryUsage()

  const status = dbCheck.status === "up" ? "healthy" : "unhealthy"

  return c.json(
    {
      status,
      timestamp: new Date().toISOString(),
      uptime: Math.floor((Date.now() - startTime) / 1000),
      checks: {
        database: dbCheck,
        memory,
      },
    },
    status === "healthy" ? 200 : 503
  )
})

// Readiness check (Kubernetes)
healthRoutes.get("/health/ready", async (c) => {
  const dbCheck = await checkDatabase()

  if (dbCheck.status === "up") {
    return c.json({ ready: true })
  }

  return c.json({ ready: false, reason: "Database not ready" }, 503)
})

// Liveness check (Kubernetes)
healthRoutes.get("/health/live", (c) => {
  return c.json({ alive: true })
})

async function checkDatabase() {
  const start = performance.now()

  try {
    await db.execute(sql`SELECT 1`)
    return {
      status: "up" as const,
      latency: Math.round(performance.now() - start),
    }
  } catch (error) {
    return {
      status: "down" as const,
      error: error instanceof Error ? error.message : "Unknown error",
    }
  }
}

function getMemoryUsage() {
  const usage = process.memoryUsage()
  return {
    heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024)}MB`,
    heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024)}MB`,
    rss: `${Math.round(usage.rss / 1024 / 1024)}MB`,
  }
}
