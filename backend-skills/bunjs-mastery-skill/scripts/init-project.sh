#!/bin/bash

# ==============================================
# Bun.js Project Initializer
# Senior Developer Edition
# ==============================================

set -e

PROJECT_NAME=${1:-"my-bun-app"}

echo "🚀 Creating project: $PROJECT_NAME"

# Create project directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Initialize bun
bun init -y

# Create folder structure
echo "📁 Creating folder structure..."
mkdir -p src/{config,routes,controllers,services,repositories,middlewares,utils,types,db}
mkdir -p tests/{unit,integration,e2e}
mkdir -p docker

# Install dependencies
echo "📦 Installing dependencies..."
bun add hono zod drizzle-orm postgres pino ioredis nanoid
bun add -d @types/bun vitest drizzle-kit pino-pretty @biomejs/biome

# Create .env.example
cat > .env.example << 'EOF'
NODE_ENV=development
PORT=3000
LOG_LEVEL=debug

# Database
DATABASE_URL=postgres://postgres:postgres@localhost:5432/app_dev

# Redis
REDIS_URL=redis://localhost:6379

# Auth
JWT_SECRET=your-super-secret-key-at-least-32-chars
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
node_modules
dist
.env
.env.local
*.log
coverage
.DS_Store
EOF

# Create tsconfig.json
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "types": ["bun-types"],
    "strict": true,
    "skipLibCheck": true,
    "noEmit": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*", "tests/**/*"],
  "exclude": ["node_modules"]
}
EOF

# Update package.json scripts
cat > package.json << 'EOF'
{
  "name": "PROJECT_NAME_PLACEHOLDER",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "bun --watch src/index.ts",
    "build": "bun build src/index.ts --outdir dist --target bun",
    "start": "NODE_ENV=production bun dist/index.js",
    "test": "vitest",
    "test:unit": "vitest run --dir tests/unit",
    "test:integration": "vitest run --dir tests/integration",
    "test:coverage": "vitest run --coverage",
    "lint": "bunx @biomejs/biome check .",
    "format": "bunx @biomejs/biome format --write .",
    "db:generate": "drizzle-kit generate",
    "db:migrate": "drizzle-kit migrate",
    "db:studio": "drizzle-kit studio",
    "docker:dev": "docker compose -f docker/docker-compose.dev.yml up",
    "docker:build": "docker build -t PROJECT_NAME_PLACEHOLDER .",
    "docker:prod": "docker compose -f docker/docker-compose.yml up -d"
  },
  "dependencies": {
    "drizzle-orm": "latest",
    "hono": "latest",
    "ioredis": "latest",
    "nanoid": "latest",
    "pino": "latest",
    "postgres": "latest",
    "zod": "latest"
  },
  "devDependencies": {
    "@biomejs/biome": "latest",
    "@types/bun": "latest",
    "drizzle-kit": "latest",
    "pino-pretty": "latest",
    "vitest": "latest"
  }
}
EOF

sed -i "s/PROJECT_NAME_PLACEHOLDER/$PROJECT_NAME/g" package.json

echo "✅ Project $PROJECT_NAME created successfully!"
echo ""
echo "Next steps:"
echo "  cd $PROJECT_NAME"
echo "  cp .env.example .env"
echo "  bun run dev"
