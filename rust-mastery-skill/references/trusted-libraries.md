# Trusted Rust Libraries Reference

Library yang sudah battle-tested di production dengan jutaan downloads.

## Table of Contents

1. [Async Runtime](#async-runtime)
2. [Web Framework](#web-framework)
3. [Database](#database)
4. [Serialization](#serialization)
5. [Error Handling](#error-handling)
6. [Observability](#observability)
7. [Testing](#testing)
8. [Security & Crypto](#security--crypto)
9. [CLI](#cli)
10. [Utilities](#utilities)

---

## Async Runtime

### tokio
**The** async runtime. No alternatives needed for 99% cases.

```toml
tokio = { version = "1", features = ["full"] }
```

```rust
#[tokio::main]
async fn main() {
    // Spawn concurrent tasks
    let handle = tokio::spawn(async {
        // Background work
    });
    
    // Timeouts
    tokio::time::timeout(Duration::from_secs(5), async_operation()).await?;
    
    // Intervals
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    loop {
        interval.tick().await;
        perform_periodic_task().await;
    }
}
```

### async-trait
Async methods in traits (sampai Rust native support).

```rust
#[async_trait]
trait Repository {
    async fn find(&self, id: Uuid) -> Result<Entity>;
    async fn save(&self, entity: &Entity) -> Result<()>;
}
```

---

## Web Framework

### axum (Recommended)
Modern, ergonomic, built on Tower ecosystem.

```rust
use axum::{
    Router, Json, Extension,
    extract::{Path, State, Query},
    http::StatusCode,
    response::IntoResponse,
};

async fn get_user(
    State(pool): State<PgPool>,
    Path(id): Path<Uuid>,
) -> Result<Json<User>, AppError> {
    let user = sqlx::query_as!(User, "SELECT * FROM users WHERE id = $1", id)
        .fetch_one(&pool)
        .await?;
    Ok(Json(user))
}

let app = Router::new()
    .route("/users/:id", get(get_user))
    .route("/users", post(create_user))
    .layer(TraceLayer::new_for_http())
    .with_state(pool);
```

### tower & tower-http
Middleware ecosystem.

```rust
use tower_http::{
    trace::TraceLayer,
    cors::CorsLayer,
    compression::CompressionLayer,
    timeout::TimeoutLayer,
};

let app = Router::new()
    .route("/api", get(handler))
    .layer(
        ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(TimeoutLayer::new(Duration::from_secs(30)))
            .layer(CompressionLayer::new())
            .layer(CorsLayer::permissive())
    );
```

### reqwest
HTTP client.

```rust
let client = reqwest::Client::builder()
    .timeout(Duration::from_secs(10))
    .pool_max_idle_per_host(10)
    .build()?;

let response = client
    .post("https://api.example.com/data")
    .bearer_auth(&token)
    .json(&payload)
    .send()
    .await?
    .error_for_status()?
    .json::<Response>()
    .await?;
```

---

## Database

### sqlx (Recommended)
Compile-time checked SQL. Zero-cost abstraction.

```rust
// Compile-time verified query
let users = sqlx::query_as!(
    User,
    r#"
    SELECT id, name, email, created_at
    FROM users
    WHERE active = true
    ORDER BY created_at DESC
    LIMIT $1
    "#,
    limit
)
.fetch_all(&pool)
.await?;

// Transaction
let mut tx = pool.begin().await?;
sqlx::query!("INSERT INTO users (name) VALUES ($1)", name)
    .execute(&mut *tx)
    .await?;
sqlx::query!("INSERT INTO audit_log (action) VALUES ($1)", "user_created")
    .execute(&mut *tx)
    .await?;
tx.commit().await?;
```

### deadpool-postgres / bb8
Connection pooling.

```rust
let pool = PgPoolOptions::new()
    .max_connections(20)
    .min_connections(5)
    .acquire_timeout(Duration::from_secs(5))
    .idle_timeout(Duration::from_secs(600))
    .connect(&database_url)
    .await?;
```

### redis
Redis client with connection pooling.

```rust
use redis::AsyncCommands;

let client = redis::Client::open("redis://127.0.0.1/")?;
let mut con = client.get_multiplexed_async_connection().await?;

// Set with expiry
con.set_ex::<_, _, ()>("key", "value", 3600).await?;

// Get
let value: Option<String> = con.get("key").await?;
```

---

## Serialization

### serde + serde_json
De facto standard.

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserResponse {
    pub id: Uuid,
    pub full_name: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    
    #[serde(default)]
    pub is_active: bool,
    
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: DateTime<Utc>,
}
```

### toml / config
Configuration management.

```rust
use config::{Config, Environment, File};

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub database: DatabaseSettings,
    pub server: ServerSettings,
}

impl Settings {
    pub fn new() -> Result<Self, config::ConfigError> {
        Config::builder()
            .add_source(File::with_name("config/default"))
            .add_source(File::with_name("config/local").required(false))
            .add_source(Environment::with_prefix("APP").separator("__"))
            .build()?
            .try_deserialize()
    }
}
```

---

## Error Handling

### thiserror
Derive macro untuk custom errors.

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Resource not found: {resource_type} with id {id}")]
    NotFound { resource_type: String, id: String },
    
    #[error("Unauthorized access")]
    Unauthorized,
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error(transparent)]
    Database(#[from] sqlx::Error),
    
    #[error(transparent)]
    Unexpected(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::NotFound { .. } => (StatusCode::NOT_FOUND, self.to_string()),
            AppError::Unauthorized => (StatusCode::UNAUTHORIZED, self.to_string()),
            AppError::Validation(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".into()),
        };
        
        (status, Json(json!({ "error": message }))).into_response()
    }
}
```

### anyhow
Untuk application code yang tidak perlu typed errors.

```rust
use anyhow::{Context, Result, bail, ensure};

fn process_file(path: &Path) -> Result<Data> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read configuration file")?;
    
    ensure!(!content.is_empty(), "Configuration file is empty");
    
    let data: Data = serde_json::from_str(&content)
        .context("Failed to parse configuration")?;
    
    if !data.is_valid() {
        bail!("Invalid configuration: missing required fields");
    }
    
    Ok(data)
}
```

---

## Observability

### tracing
Structured logging + distributed tracing.

```rust
use tracing::{info, warn, error, instrument, span, Level};

#[instrument(skip(pool), fields(user_id = %id))]
async fn get_user(pool: &PgPool, id: Uuid) -> Result<User> {
    info!("Fetching user from database");
    
    let user = sqlx::query_as!(User, "SELECT * FROM users WHERE id = $1", id)
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            error!(error = %e, "Database query failed");
            e
        })?;
    
    match user {
        Some(u) => {
            info!(name = %u.name, "User found");
            Ok(u)
        }
        None => {
            warn!("User not found");
            Err(AppError::NotFound { id })
        }
    }
}
```

### tracing-subscriber
Subscriber configuration.

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,sqlx=warn,tower_http=debug"));

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().json()) // JSON for production
        .init();
}
```

### metrics + metrics-exporter-prometheus
Application metrics.

```rust
use metrics::{counter, gauge, histogram};

counter!("http_requests_total", "method" => "GET", "path" => "/users").increment(1);
gauge!("active_connections").set(42.0);
histogram!("request_duration_seconds").record(0.025);
```

---

## Testing

### rstest
Parameterized tests.

```rust
use rstest::*;

#[fixture]
fn user() -> User {
    User::new("test@example.com")
}

#[rstest]
#[case("", false)]
#[case("invalid", false)]
#[case("valid@email.com", true)]
fn test_email_validation(#[case] email: &str, #[case] expected: bool) {
    assert_eq!(is_valid_email(email), expected);
}

#[rstest]
fn test_user_creation(user: User) {
    assert!(user.is_valid());
}
```

### wiremock
HTTP mocking.

```rust
use wiremock::{MockServer, Mock, ResponseTemplate, matchers::*};

#[tokio::test]
async fn test_external_api_call() {
    let mock_server = MockServer::start().await;
    
    Mock::given(method("GET"))
        .and(path("/api/users/123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "123",
            "name": "Test User"
        })))
        .mount(&mock_server)
        .await;
    
    let client = ApiClient::new(&mock_server.uri());
    let user = client.get_user("123").await.unwrap();
    
    assert_eq!(user.name, "Test User");
}
```

### fake
Generate fake data.

```rust
use fake::{Fake, Faker};
use fake::faker::name::en::*;
use fake::faker::internet::en::*;

let name: String = Name().fake();
let email: String = SafeEmail().fake();
let user: User = Faker.fake(); // With derive
```

---

## Security & Crypto

### argon2
Password hashing (winner of PHC).

```rust
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::SaltString;

fn hash_password(password: &str) -> Result<String> {
    let salt = SaltString::generate(&mut rand::thread_rng());
    let argon2 = Argon2::default();
    let hash = argon2.hash_password(password.as_bytes(), &salt)?.to_string();
    Ok(hash)
}

fn verify_password(password: &str, hash: &str) -> Result<bool> {
    let parsed_hash = PasswordHash::new(hash)?;
    Ok(Argon2::default().verify_password(password.as_bytes(), &parsed_hash).is_ok())
}
```

### jsonwebtoken
JWT handling.

```rust
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
    iat: usize,
}

fn create_token(user_id: &str, secret: &[u8]) -> Result<String> {
    let now = Utc::now();
    let claims = Claims {
        sub: user_id.to_owned(),
        iat: now.timestamp() as usize,
        exp: (now + Duration::hours(24)).timestamp() as usize,
    };
    
    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret))
        .context("Failed to create token")
}
```

### secrecy
Prevent accidental secret logging.

```rust
use secrecy::{Secret, ExposeSecret};

struct Config {
    database_password: Secret<String>,
    api_key: Secret<String>,
}

// Tidak akan ter-print/log secara tidak sengaja
impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("database_password", &"[REDACTED]")
            .finish()
    }
}

// Akses explicit
let password = config.database_password.expose_secret();
```

---

## CLI

### clap
Command-line parsing.

```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "myapp", version, about)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
    
    /// Enable verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the server
    Serve {
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
    /// Run database migrations
    Migrate,
}
```

---

## Utilities

### uuid
UUID generation.

```rust
use uuid::Uuid;

let id = Uuid::new_v4();  // Random UUID
let id = Uuid::now_v7();  // Time-ordered UUID (recommended for DB primary keys)
```

### chrono
Date/time handling.

```rust
use chrono::{DateTime, Utc, Duration};

let now: DateTime<Utc> = Utc::now();
let tomorrow = now + Duration::days(1);
let formatted = now.format("%Y-%m-%d %H:%M:%S").to_string();
```

### once_cell / std::sync::OnceLock
Lazy static initialization (OnceLock sudah di std sejak 1.70).

```rust
use std::sync::OnceLock;

static CONFIG: OnceLock<Config> = OnceLock::new();

fn get_config() -> &'static Config {
    CONFIG.get_or_init(|| Config::load().expect("Failed to load config"))
}
```

### dashmap
Concurrent HashMap.

```rust
use dashmap::DashMap;

let cache: DashMap<String, User> = DashMap::new();
cache.insert("key".to_string(), user);

if let Some(user) = cache.get("key") {
    println!("Found: {:?}", *user);
}
```

### parking_lot
Faster Mutex/RwLock implementation.

```rust
use parking_lot::{Mutex, RwLock};

let data = RwLock::new(vec![1, 2, 3]);

// Multiple readers
{
    let read = data.read();
    println!("{:?}", *read);
}

// Single writer
{
    let mut write = data.write();
    write.push(4);
}
```

---

## Libraries to AVOID

| Library | Issue | Alternative |
|---------|-------|-------------|
| `actix-web` | Over-engineered, actor model complexity | `axum` |
| `diesel` | Compile times, less flexible than raw SQL | `sqlx` |
| `rocket` | Slow development, proc macro heavy | `axum` |
| `failure` | Deprecated | `thiserror` + `anyhow` |
| `log` + `env_logger` | Less powerful | `tracing` |
| `lazy_static` | Deprecated | `std::sync::OnceLock` |
