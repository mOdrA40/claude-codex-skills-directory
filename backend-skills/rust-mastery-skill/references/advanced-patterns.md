# Advanced Patterns

Pattern-pattern advanced yang sering digunakan senior engineer.

## Table of Contents

1. [Builder Pattern](#builder-pattern)
2. [Type-State Pattern](#type-state-pattern)
3. [Newtype Pattern](#newtype-pattern)
4. [Repository Pattern](#repository-pattern)
5. [Dependency Injection](#dependency-injection)
6. [Error Handling Patterns](#error-handling-patterns)
7. [Async Patterns](#async-patterns)

---

## Builder Pattern

Untuk konstruksi object yang kompleks dengan banyak optional fields.

```rust
#[derive(Debug)]
pub struct HttpClient {
    base_url: String,
    timeout: Duration,
    max_retries: u32,
    headers: HeaderMap,
}

#[derive(Default)]
pub struct HttpClientBuilder {
    base_url: Option<String>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
    headers: HeaderMap,
}

impl HttpClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
    
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }
    
    pub fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(
            key.parse().unwrap(),
            value.parse().unwrap(),
        );
        self
    }
    
    pub fn build(self) -> Result<HttpClient, BuilderError> {
        let base_url = self.base_url
            .ok_or(BuilderError::MissingField("base_url"))?;
        
        Ok(HttpClient {
            base_url,
            timeout: self.timeout.unwrap_or(Duration::from_secs(30)),
            max_retries: self.max_retries.unwrap_or(3),
            headers: self.headers,
        })
    }
}

// Usage
let client = HttpClientBuilder::new()
    .base_url("https://api.example.com")
    .timeout(Duration::from_secs(10))
    .header("Authorization", "Bearer token")
    .build()?;
```

### derive_builder Crate (Simpler)

```rust
use derive_builder::Builder;

#[derive(Builder, Debug)]
#[builder(setter(into))]
pub struct HttpClient {
    base_url: String,
    #[builder(default = "Duration::from_secs(30)")]
    timeout: Duration,
    #[builder(default = "3")]
    max_retries: u32,
}

let client = HttpClientBuilder::default()
    .base_url("https://api.example.com")
    .build()?;
```

---

## Type-State Pattern

Compile-time enforcement of valid state transitions.

```rust
// State markers (zero-sized types)
pub struct Draft;
pub struct Submitted;
pub struct Approved;
pub struct Rejected;

pub struct Document<State> {
    id: Uuid,
    content: String,
    _state: PhantomData<State>,
}

impl Document<Draft> {
    pub fn new(content: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            _state: PhantomData,
        }
    }
    
    pub fn edit(&mut self, content: String) {
        self.content = content;
    }
    
    pub fn submit(self) -> Document<Submitted> {
        Document {
            id: self.id,
            content: self.content,
            _state: PhantomData,
        }
    }
}

impl Document<Submitted> {
    pub fn approve(self) -> Document<Approved> {
        Document {
            id: self.id,
            content: self.content,
            _state: PhantomData,
        }
    }
    
    pub fn reject(self) -> Document<Rejected> {
        Document {
            id: self.id,
            content: self.content,
            _state: PhantomData,
        }
    }
}

impl Document<Rejected> {
    pub fn revise(self, content: String) -> Document<Draft> {
        Document {
            id: self.id,
            content,
            _state: PhantomData,
        }
    }
}

// Usage - Compiler enforces valid transitions!
let doc = Document::new("Initial content".into());
// doc.approve(); // ERROR: Not available for Draft

let submitted = doc.submit();
// submitted.edit("new"); // ERROR: Not available for Submitted

let approved = submitted.approve();
// Now we have Document<Approved>
```

---

## Newtype Pattern

Type safety melalui wrapper types.

```rust
// ❌ BAD: Mudah tertukar parameter
fn create_user(email: String, password: String) { }
create_user(password, email); // Compiles but WRONG!

// ✅ GOOD: Newtype prevents mistakes
#[derive(Debug, Clone)]
pub struct Email(String);

impl Email {
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        if !value.contains('@') {
            return Err(ValidationError::InvalidEmail);
        }
        Ok(Self(value))
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct HashedPassword(String);

impl HashedPassword {
    pub fn new(plain: &str) -> Result<Self, HashError> {
        let hashed = hash_password(plain)?;
        Ok(Self(hashed))
    }
}

// Now impossible to mix up!
fn create_user(email: Email, password: HashedPassword) { }
// create_user(password, email); // Compiler ERROR!
```

### Derive Newtype dengan derive_more

```rust
use derive_more::{Display, From, Into, AsRef};

#[derive(Debug, Clone, Display, From, Into, AsRef)]
pub struct UserId(Uuid);

#[derive(Debug, Clone, Display, From, Into, AsRef)]
pub struct OrderId(Uuid);

// UserId dan OrderId adalah tipe berbeda
fn process_order(user_id: UserId, order_id: OrderId) { }
```

---

## Repository Pattern

Abstraksi akses data.

```rust
use async_trait::async_trait;

// Domain entity
#[derive(Debug, Clone)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub name: String,
}

// Repository trait (domain layer)
#[async_trait]
pub trait UserRepository: Send + Sync {
    async fn find_by_id(&self, id: Uuid) -> Result<Option<User>>;
    async fn find_by_email(&self, email: &str) -> Result<Option<User>>;
    async fn save(&self, user: &User) -> Result<()>;
    async fn delete(&self, id: Uuid) -> Result<()>;
}

// PostgreSQL implementation (infrastructure layer)
pub struct PgUserRepository {
    pool: PgPool,
}

impl PgUserRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl UserRepository for PgUserRepository {
    async fn find_by_id(&self, id: Uuid) -> Result<Option<User>> {
        sqlx::query_as!(
            User,
            "SELECT id, email, name FROM users WHERE id = $1",
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(Into::into)
    }
    
    async fn find_by_email(&self, email: &str) -> Result<Option<User>> {
        sqlx::query_as!(
            User,
            "SELECT id, email, name FROM users WHERE email = $1",
            email
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(Into::into)
    }
    
    async fn save(&self, user: &User) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO users (id, email, name)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO UPDATE SET email = $2, name = $3
            "#,
            user.id, user.email, user.name
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }
    
    async fn delete(&self, id: Uuid) -> Result<()> {
        sqlx::query!("DELETE FROM users WHERE id = $1", id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

// Mock implementation for testing
#[cfg(test)]
pub struct MockUserRepository {
    users: std::sync::Mutex<HashMap<Uuid, User>>,
}

#[cfg(test)]
#[async_trait]
impl UserRepository for MockUserRepository {
    async fn find_by_id(&self, id: Uuid) -> Result<Option<User>> {
        Ok(self.users.lock().unwrap().get(&id).cloned())
    }
    // ... other methods
}
```

---

## Dependency Injection

Constructor injection tanpa framework.

```rust
// Service dengan dependencies
pub struct UserService<R: UserRepository, N: NotificationService> {
    user_repo: R,
    notifier: N,
}

impl<R: UserRepository, N: NotificationService> UserService<R, N> {
    pub fn new(user_repo: R, notifier: N) -> Self {
        Self { user_repo, notifier }
    }
    
    pub async fn register(&self, email: &str, name: &str) -> Result<User> {
        let user = User {
            id: Uuid::new_v4(),
            email: email.to_string(),
            name: name.to_string(),
        };
        
        self.user_repo.save(&user).await?;
        self.notifier.send_welcome(&user).await?;
        
        Ok(user)
    }
}

// Type alias untuk production
pub type ProdUserService = UserService<PgUserRepository, EmailNotifier>;

// Factory function
pub fn create_user_service(pool: PgPool, smtp: SmtpConfig) -> ProdUserService {
    let user_repo = PgUserRepository::new(pool);
    let notifier = EmailNotifier::new(smtp);
    UserService::new(user_repo, notifier)
}

// Untuk testing
#[cfg(test)]
mod tests {
    fn create_test_service() -> UserService<MockUserRepository, MockNotifier> {
        UserService::new(
            MockUserRepository::new(),
            MockNotifier::new(),
        )
    }
}
```

### Arc untuk Shared State

```rust
use std::sync::Arc;

// Application state
pub struct AppState {
    pub user_service: Arc<dyn UserRepository>,
    pub config: Arc<Config>,
}

// Setup
let pool = PgPool::connect(&database_url).await?;
let user_repo = Arc::new(PgUserRepository::new(pool));

let state = AppState {
    user_service: user_repo,
    config: Arc::new(config),
};

// Axum handler dengan state
async fn get_user(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<User>> {
    let user = state.user_service.find_by_id(id).await?;
    Ok(Json(user.ok_or(AppError::NotFound)?))
}
```

---

## Error Handling Patterns

### Domain Error + API Error Separation

```rust
// Domain errors (business logic)
#[derive(Error, Debug)]
pub enum DomainError {
    #[error("User not found: {0}")]
    UserNotFound(Uuid),
    
    #[error("Email already registered")]
    EmailExists,
    
    #[error("Invalid email format")]
    InvalidEmail,
    
    #[error("Password too weak")]
    WeakPassword,
}

// Infrastructure errors
#[derive(Error, Debug)]
pub enum InfraError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Cache error: {0}")]
    Cache(#[from] redis::RedisError),
    
    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),
}

// Application error (combines both)
#[derive(Error, Debug)]
pub enum AppError {
    #[error(transparent)]
    Domain(#[from] DomainError),
    
    #[error(transparent)]
    Infrastructure(#[from] InfraError),
    
    #[error("Unexpected error: {0}")]
    Unexpected(#[from] anyhow::Error),
}

// API response mapping
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::Domain(e) => match e {
                DomainError::UserNotFound(_) => (StatusCode::NOT_FOUND, e.to_string()),
                DomainError::EmailExists => (StatusCode::CONFLICT, e.to_string()),
                DomainError::InvalidEmail |
                DomainError::WeakPassword => (StatusCode::BAD_REQUEST, e.to_string()),
            },
            AppError::Infrastructure(_) => {
                // Log internal error, return generic message
                error!(error = %self, "Infrastructure error");
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".to_string())
            }
            AppError::Unexpected(_) => {
                error!(error = %self, "Unexpected error");
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".to_string())
            }
        };
        
        (status, Json(json!({ "error": message }))).into_response()
    }
}
```

---

## Async Patterns

### Graceful Shutdown

```rust
use tokio::signal;
use tokio_util::sync::CancellationToken;

async fn run_server(cancel_token: CancellationToken) {
    let app = create_router();
    
    let listener = TcpListener::bind("0.0.0.0:8080").await.unwrap();
    
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel_token.cancelled().await;
            info!("Shutdown signal received");
        })
        .await
        .unwrap();
}

#[tokio::main]
async fn main() {
    let cancel_token = CancellationToken::new();
    
    // Spawn server
    let server_token = cancel_token.clone();
    let server_handle = tokio::spawn(run_server(server_token));
    
    // Spawn background workers
    let worker_token = cancel_token.clone();
    let worker_handle = tokio::spawn(run_background_worker(worker_token));
    
    // Wait for shutdown signal
    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("Received Ctrl+C");
        }
        _ = terminate_signal() => {
            info!("Received SIGTERM");
        }
    }
    
    // Trigger shutdown
    cancel_token.cancel();
    
    // Wait for tasks to complete
    let _ = tokio::join!(server_handle, worker_handle);
    info!("Shutdown complete");
}

#[cfg(unix)]
async fn terminate_signal() {
    signal::unix::signal(signal::unix::SignalKind::terminate())
        .expect("Failed to create SIGTERM handler")
        .recv()
        .await;
}
```

### Rate Limiting dengan tower

```rust
use tower::ServiceBuilder;
use tower_governor::{Governor, GovernorConfigBuilder};

let governor_conf = GovernorConfigBuilder::default()
    .per_second(10)
    .burst_size(20)
    .finish()
    .unwrap();

let app = Router::new()
    .route("/api/data", get(handler))
    .layer(
        ServiceBuilder::new()
            .layer(Governor::layer(&governor_conf))
    );
```

### Retry dengan exponential backoff

```rust
use tokio_retry::{Retry, strategy::{ExponentialBackoff, jitter}};

async fn fetch_with_retry(url: &str) -> Result<Response> {
    let retry_strategy = ExponentialBackoff::from_millis(100)
        .max_delay(Duration::from_secs(10))
        .map(jitter)
        .take(5); // Max 5 retries
    
    Retry::spawn(retry_strategy, || async {
        client.get(url).send().await
    })
    .await
    .map_err(Into::into)
}
```

### Circuit Breaker

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct CircuitBreaker {
    failure_count: AtomicU32,
    last_failure: AtomicU64,
    threshold: u32,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            last_failure: AtomicU64::new(0),
            threshold,
            reset_timeout,
        }
    }
    
    pub fn is_open(&self) -> bool {
        let failures = self.failure_count.load(Ordering::SeqCst);
        if failures < self.threshold {
            return false;
        }
        
        let last = self.last_failure.load(Ordering::SeqCst);
        let now = Instant::now().elapsed().as_secs();
        
        now - last < self.reset_timeout.as_secs()
    }
    
    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::SeqCst);
    }
    
    pub fn record_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::SeqCst);
        self.last_failure.store(
            Instant::now().elapsed().as_secs(),
            Ordering::SeqCst,
        );
    }
    
    pub async fn call<F, T, E>(&self, f: F) -> Result<T, CircuitError<E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        if self.is_open() {
            return Err(CircuitError::Open);
        }
        
        match f.await {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(e) => {
                self.record_failure();
                Err(CircuitError::Inner(e))
            }
        }
    }
}
```

---

## When NOT to Use These Patterns

Senior engineer tahu kapan pattern TIDAK diperlukan:

1. **Builder Pattern** - Skip jika hanya 2-3 required fields
2. **Type-State** - Skip jika state transitions simple
3. **Repository** - Skip untuk small projects tanpa testing
4. **DI Framework** - Rust tidak butuh, constructor injection cukup

**Rule: Complexity harus justified by real needs, bukan hypothetical future requirements.**
