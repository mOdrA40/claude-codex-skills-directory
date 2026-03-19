/**
 * Health Check Endpoint Template
 * 
 * Provides comprehensive health status for:
 * - Application status
 * - Database connectivity
 * - Redis connectivity
 * - Memory usage
 */

import { Hono } from "hono"
import { db } from "../db"
import { redis } from "../config/redis"
import { sql } from "drizzle-orm"

interface HealthStatus {
  status: "healthy" | "unhealthy" | "degraded"
  timestamp: string
  uptime: number
  version: string
  checks: {
    database: CheckResult
    redis: CheckResult
    memory: MemoryStatus
  }
}

interface CheckResult {
  status: "up" | "down"
  latency?: number
  error?: string
}

interface MemoryStatus {
  heapUsed: string
  heapTotal: string
  rss: string
  percentage: number
}

const startTime = Date.now()

export const healthRoutes = new Hono()

// Simple health check (untuk load balancer)
healthRoutes.get("/health", (c) => {
  return c.json({ status: "ok" })
})

// Detailed health check (untuk monitoring)
healthRoutes.get("/health/detailed", async (c) => {
  const checks = await Promise.all([
    checkDatabase(),
    checkRedis(),
    getMemoryStatus(),
  ])

  const [database, redisCheck, memory] = checks

  const allUp = database.status === "up" && redisCheck.status === "up"
  const someUp = database.status === "up" || redisCheck.status === "up"

  const status: HealthStatus = {
    status: allUp ? "healthy" : someUp ? "degraded" : "unhealthy",
    timestamp: new Date().toISOString(),
    uptime: Math.floor((Date.now() - startTime) / 1000),
    version: process.env.npm_package_version || "1.0.0",
    checks: {
      database,
      redis: redisCheck,
      memory,
    },
  }

  const httpStatus = status.status === "healthy" ? 200 : status.status === "degraded" ? 200 : 503

  return c.json(status, httpStatus)
})

// Readiness check (untuk Kubernetes)
healthRoutes.get("/health/ready", async (c) => {
  const dbCheck = await checkDatabase()
  
  if (dbCheck.status === "up") {
    return c.json({ ready: true })
  }
  
  return c.json({ ready: false, reason: "Database not ready" }, 503)
})

// Liveness check (untuk Kubernetes)
healthRoutes.get("/health/live", (c) => {
  return c.json({ alive: true })
})

async function checkDatabase(): Promise<CheckResult> {
  const start = performance.now()
  
  try {
    await db.execute(sql`SELECT 1`)
    return {
      status: "up",
      latency: Math.round(performance.now() - start),
    }
  } catch (error) {
    return {
      status: "down",
      error: error instanceof Error ? error.message : "Unknown error",
    }
  }
}

async function checkRedis(): Promise<CheckResult> {
  const start = performance.now()
  
  try {
    await redis.ping()
    return {
      status: "up",
      latency: Math.round(performance.now() - start),
    }
  } catch (error) {
    return {
      status: "down",
      error: error instanceof Error ? error.message : "Unknown error",
    }
  }
}

function getMemoryStatus(): MemoryStatus {
  const usage = process.memoryUsage()
  const heapPercentage = (usage.heapUsed / usage.heapTotal) * 100
  
  return {
    heapUsed: `${Math.round(usage.heapUsed / 1024 / 1024)}MB`,
    heapTotal: `${Math.round(usage.heapTotal / 1024 / 1024)}MB`,
    rss: `${Math.round(usage.rss / 1024 / 1024)}MB`,
    percentage: Math.round(heapPercentage),
  }
}
