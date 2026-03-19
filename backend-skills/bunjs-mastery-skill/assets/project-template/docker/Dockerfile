# ========================================
# Stage 1: Dependencies
# ========================================
FROM oven/bun:1-alpine AS deps
WORKDIR /app

COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile

# ========================================
# Stage 2: Builder
# ========================================
FROM oven/bun:1-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN bun run build
RUN bun install --production --frozen-lockfile

# ========================================
# Stage 3: Runner (Production)
# ========================================
FROM oven/bun:1-alpine AS runner
WORKDIR /app

# Security: Non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

ENV NODE_ENV=production
ENV TZ=Asia/Jakarta

# Copy built files
COPY --from=builder --chown=appuser:appgroup /app/dist ./dist
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/package.json ./

USER appuser

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

CMD ["bun", "run", "dist/index.js"]
