#!/usr/bin/env python3
"""
Rust Project Scaffolder

Creates a production-ready Rust project structure with:
- Clean architecture folder structure
- Docker multi-stage build
- docker-compose for development
- Pre-configured Cargo.toml with recommended dependencies
- Basic CI/CD templates

Usage:
    python scaffold_project.py <project-name> [--path <output-dir>]
"""

import argparse
import os
from pathlib import Path

def create_cargo_toml(project_name: str) -> str:
    return f'''[package]
name = "{project_name}"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <your@email.com>"]
description = "Project description"
license = "MIT OR Apache-2.0"

[features]
default = []

[dependencies]
# Async runtime
tokio = {{ version = "1", features = ["full"] }}

# Web framework
axum = {{ version = "0.7", features = ["macros"] }}
tower = "0.4"
tower-http = {{ version = "0.5", features = ["trace", "cors", "compression-gzip"] }}

# Serialization
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"

# Database
sqlx = {{ version = "0.7", features = ["runtime-tokio", "postgres", "uuid", "chrono"] }}

# Observability
tracing = "0.1"
tracing-subscriber = {{ version = "0.3", features = ["env-filter", "json"] }}

# Error handling
thiserror = "1"
anyhow = "1"

# Configuration
config = "0.14"

# Utilities
uuid = {{ version = "1", features = ["v4", "v7", "serde"] }}
chrono = {{ version = "0.4", features = ["serde"] }}

[dev-dependencies]
criterion = {{ version = "0.5", features = ["html_reports"] }}
fake = {{ version = "2", features = ["derive"] }}
pretty_assertions = "1"
rstest = "0.18"
tokio-test = "0.4"
wiremock = "0.6"

[[bench]]
name = "benchmarks"
harness = false

[profile.release]
lto = "thin"
codegen-units = 1
panic = "abort"
strip = true

[profile.dev.package."*"]
opt-level = 3
'''

def create_dockerfile() -> str:
    return '''# === Stage 1: Chef (Dependency Caching) ===
FROM rust:1.75-slim-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

# === Stage 2: Planner ===
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# === Stage 3: Builder ===
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --locked

# === Stage 4: Runtime ===
FROM debian:bookworm-slim AS runtime

RUN groupadd --gid 1000 appuser \\
    && useradd --uid 1000 --gid 1000 -m appuser

RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/app-name /app/app-name
RUN chown -R appuser:appuser /app

USER appuser
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
ENTRYPOINT ["/app/app-name"]
'''

def create_docker_compose() -> str:
    return '''version: "3.9"

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=debug
      - DATABASE_URL=postgres://user:pass@db:5432/dbname
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: dbname
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d dbname"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres-data:
'''

def create_main_rs() -> str:
    return '''use anyhow::Result;
use tracing::info;

mod config;
mod domain;
mod infrastructure;
mod api;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    info!("Starting application");

    // Load configuration
    let config = config::Settings::new()?;

    // Initialize infrastructure
    let pool = infrastructure::database::create_pool(&config.database).await?;

    // Run migrations
    sqlx::migrate!().run(&pool).await?;

    // Build and run API
    let app = api::create_router(pool);
    let listener = tokio::net::TcpListener::bind(&config.server.address).await?;
    
    info!("Server listening on {}", config.server.address);
    axum::serve(listener, app).await?;

    Ok(())
}
'''

def create_lib_rs() -> str:
    return '''//! Project library root
//! 
//! Re-exports public API for library usage.

pub mod config;
pub mod domain;
pub mod infrastructure;
pub mod api;
'''

def create_config_mod() -> str:
    return '''mod settings;

pub use settings::Settings;
'''

def create_config_settings() -> str:
    return '''use config::{Config, Environment, File};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub server: ServerSettings,
    pub database: DatabaseSettings,
}

#[derive(Debug, Deserialize)]
pub struct ServerSettings {
    pub address: String,
}

#[derive(Debug, Deserialize)]
pub struct DatabaseSettings {
    pub url: String,
    pub max_connections: u32,
}

impl Settings {
    pub fn new() -> Result<Self, config::ConfigError> {
        Config::builder()
            .add_source(File::with_name("config/default").required(false))
            .add_source(File::with_name("config/local").required(false))
            .add_source(Environment::with_prefix("APP").separator("__"))
            .build()?
            .try_deserialize()
    }
}
'''

def create_domain_mod() -> str:
    return '''pub mod models;
pub mod services;
pub mod errors;

pub use errors::DomainError;
'''

def create_domain_errors() -> str:
    return '''use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum DomainError {
    #[error("Entity not found: {entity_type} with id {id}")]
    NotFound { entity_type: String, id: Uuid },

    #[error("Validation error: {message}")]
    Validation { message: String },

    #[error("Conflict: {message}")]
    Conflict { message: String },
}
'''

def create_infrastructure_mod() -> str:
    return '''pub mod database;
'''

def create_infrastructure_database() -> str:
    return '''use anyhow::Result;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::time::Duration;

use crate::config::DatabaseSettings;

pub async fn create_pool(config: &DatabaseSettings) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(config.max_connections)
        .acquire_timeout(Duration::from_secs(5))
        .connect(&config.url)
        .await?;

    Ok(pool)
}
'''

def create_api_mod() -> str:
    return '''mod routes;
mod handlers;
mod middleware;
mod responses;

pub use routes::create_router;
'''

def create_api_routes() -> str:
    return '''use axum::{Router, routing::get};
use sqlx::PgPool;
use tower_http::trace::TraceLayer;

use super::handlers;

pub fn create_router(pool: PgPool) -> Router {
    Router::new()
        .route("/health", get(handlers::health_check))
        .layer(TraceLayer::new_for_http())
        .with_state(pool)
}
'''

def create_api_handlers() -> str:
    return '''use axum::Json;
use serde_json::{json, Value};

pub async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION")
    }))
}
'''

def create_gitignore() -> str:
    return '''/target
.env
*.log
*.pdb
.DS_Store
Thumbs.db
.idea/
.vscode/
*.swp
*.swo
'''

def create_env_example() -> str:
    return '''# Server
APP_SERVER__ADDRESS=0.0.0.0:8080

# Database
APP_DATABASE__URL=postgres://user:pass@localhost:5432/dbname
APP_DATABASE__MAX_CONNECTIONS=10

# Logging
RUST_LOG=info,sqlx=warn
'''

def create_justfile(project_name: str) -> str:
    return f'''# Development commands

# Run development server with auto-reload
dev:
    cargo watch -x 'run'

# Run tests
test:
    cargo test

# Run tests with output
test-verbose:
    cargo test -- --nocapture

# Run clippy with strict warnings
lint:
    cargo clippy -- -D warnings -W clippy::pedantic

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt --check

# Security audit
audit:
    cargo audit

# Build release
build:
    cargo build --release

# Run benchmarks
bench:
    cargo bench

# Generate documentation
docs:
    cargo doc --open

# Docker build
docker-build:
    docker build -t {project_name}:latest .

# Docker compose up
up:
    docker compose up -d

# Docker compose down
down:
    docker compose down

# Docker logs
logs:
    docker compose logs -f app

# Clean build artifacts
clean:
    cargo clean

# Full check (lint + test + fmt)
check: fmt-check lint test
    @echo "All checks passed!"
'''

def scaffold_project(project_name: str, output_path: Path):
    """Create the full project structure."""
    
    root = output_path / project_name
    
    # Create directories
    dirs = [
        root / "src" / "config",
        root / "src" / "domain" / "models",
        root / "src" / "domain" / "services",
        root / "src" / "infrastructure" / "database",
        root / "src" / "api" / "handlers",
        root / "src" / "api" / "middleware",
        root / "tests" / "common",
        root / "benches",
        root / "migrations",
        root / "config",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create files
    files = {
        root / "Cargo.toml": create_cargo_toml(project_name),
        root / "Dockerfile": create_dockerfile().replace("app-name", project_name),
        root / "docker-compose.yml": create_docker_compose(),
        root / ".gitignore": create_gitignore(),
        root / ".env.example": create_env_example(),
        root / "justfile": create_justfile(project_name),
        root / "src" / "main.rs": create_main_rs(),
        root / "src" / "lib.rs": create_lib_rs(),
        root / "src" / "config" / "mod.rs": create_config_mod(),
        root / "src" / "config" / "settings.rs": create_config_settings(),
        root / "src" / "domain" / "mod.rs": create_domain_mod(),
        root / "src" / "domain" / "errors.rs": create_domain_errors(),
        root / "src" / "domain" / "models" / "mod.rs": "// Domain models\n",
        root / "src" / "domain" / "services" / "mod.rs": "// Domain services\n",
        root / "src" / "infrastructure" / "mod.rs": create_infrastructure_mod(),
        root / "src" / "infrastructure" / "database" / "mod.rs": create_infrastructure_database(),
        root / "src" / "api" / "mod.rs": create_api_mod(),
        root / "src" / "api" / "routes.rs": create_api_routes(),
        root / "src" / "api" / "handlers" / "mod.rs": create_api_handlers(),
        root / "src" / "api" / "middleware" / "mod.rs": "// Custom middleware\n",
        root / "src" / "api" / "responses.rs": "// Response types\n",
        root / "tests" / "common" / "mod.rs": "// Shared test utilities\n",
        root / "benches" / "benchmarks.rs": "// Benchmarks\n",
    }
    
    for path, content in files.items():
        path.write_text(content)
    
    print(f"✅ Created project: {root}")
    print(f"\nNext steps:")
    print(f"  cd {project_name}")
    print(f"  cargo build")
    print(f"  cargo run")

def main():
    parser = argparse.ArgumentParser(description="Scaffold a Rust project")
    parser.add_argument("project_name", help="Name of the project")
    parser.add_argument("--path", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    output_path = Path(args.path).resolve()
    scaffold_project(args.project_name, output_path)

if __name__ == "__main__":
    main()
