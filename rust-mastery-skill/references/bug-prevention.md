# Bug Prevention & Crash Patterns

Panduan untuk mengidentifikasi dan mencegah bug sebelum terjadi.

## Table of Contents

1. [Memory & Ownership Pitfalls](#memory--ownership-pitfalls)
2. [Async Pitfalls](#async-pitfalls)
3. [Error Handling Anti-Patterns](#error-handling-anti-patterns)
4. [Database Pitfalls](#database-pitfalls)
5. [Concurrency Issues](#concurrency-issues)
6. [Security Vulnerabilities](#security-vulnerabilities)
7. [Performance Killers](#performance-killers)

---

## Memory & Ownership Pitfalls

### 1. String Allocation in Hot Paths

```rust
// ❌ BAD: Allocates new String setiap call
fn format_key(prefix: &str, id: u64) -> String {
    format!("{}:{}", prefix, id)
}

// ✅ GOOD: Pre-allocate atau gunakan stack buffer
fn format_key(prefix: &str, id: u64, buf: &mut String) {
    buf.clear();
    use std::fmt::Write;
    write!(buf, "{}:{}", prefix, id).unwrap();
}

// ✅ BETTER: Gunakan compact_str untuk short strings
use compact_str::CompactString;
```

### 2. Vec Growing Overhead

```rust
// ❌ BAD: Vec grows multiple times
fn collect_items(count: usize) -> Vec<Item> {
    let mut items = Vec::new();
    for i in 0..count {
        items.push(create_item(i));
    }
    items
}

// ✅ GOOD: Pre-allocate
fn collect_items(count: usize) -> Vec<Item> {
    let mut items = Vec::with_capacity(count);
    for i in 0..count {
        items.push(create_item(i));
    }
    items
}

// ✅ ALSO GOOD: Iterator collect dengan size hint
fn collect_items(count: usize) -> Vec<Item> {
    (0..count).map(create_item).collect()
}
```

### 3. Clone Abuse

```rust
// ❌ BAD: Unnecessary clones
fn process(data: &Data) {
    let cloned = data.clone(); // Why?
    inner_process(cloned);
}

// ✅ GOOD: Pass reference atau use Cow
use std::borrow::Cow;

fn process(data: Cow<'_, Data>) {
    if needs_modification(&data) {
        let mut owned = data.into_owned();
        modify(&mut owned);
    }
}
```

### 4. Use-After-Move (Compile-time caught, but confusing)

```rust
// ❌ CONFUSING: Variable moved then "used"
let data = get_data();
send(data);
// data moved, but code below tries to use it
// log!("Sent: {:?}", data); // Compile error

// ✅ CLEAR: Clone explicitly if needed
let data = get_data();
let data_copy = data.clone();
send(data);
log!("Sent: {:?}", data_copy);
```

---

## Async Pitfalls

### 1. Blocking in Async Context

```rust
// ❌ BAD: Blocks the async runtime
async fn read_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(Into::into)
}

// ✅ GOOD: Use async version
async fn read_file(path: &Path) -> Result<String> {
    tokio::fs::read_to_string(path).await.map_err(Into::into)
}

// ✅ GOOD: Use spawn_blocking for CPU-bound work
async fn compute_hash(data: Vec<u8>) -> Result<String> {
    tokio::task::spawn_blocking(move || {
        expensive_hash_computation(&data)
    }).await?
}
```

### 2. Forgetting to .await

```rust
// ❌ BAD: Future never polled (compiler warns, but easy to miss)
async fn save_user(user: &User) {
    db.insert(user); // Missing .await!
}

// ✅ GOOD: Explicit await
async fn save_user(user: &User) -> Result<()> {
    db.insert(user).await
}

// TIP: Enable #![warn(unused_must_use)]
```

### 3. Holding Locks Across Await Points

```rust
// ❌ BAD: MutexGuard held across await - can deadlock!
async fn update_cache(cache: &Mutex<HashMap<K, V>>, key: K) {
    let mut guard = cache.lock().await;
    let value = fetch_from_db(key).await; // DEADLOCK RISK!
    guard.insert(key, value);
}

// ✅ GOOD: Minimize lock scope
async fn update_cache(cache: &Mutex<HashMap<K, V>>, key: K) {
    let value = fetch_from_db(key).await;
    let mut guard = cache.lock().await;
    guard.insert(key, value);
}

// ✅ BETTER: Use tokio::sync::Mutex for async-aware locking
```

### 4. Unbounded Channels

```rust
// ❌ BAD: Memory can grow unbounded
let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

// ✅ GOOD: Bounded with backpressure
let (tx, rx) = tokio::sync::mpsc::channel(100);

// Handle backpressure
if tx.send(item).await.is_err() {
    // Channel closed, handle gracefully
}
```

### 5. Task Leaks

```rust
// ❌ BAD: Spawned task never joined, may leak
fn start_background_task() {
    tokio::spawn(async {
        loop {
            do_work().await;
        }
    });
}

// ✅ GOOD: Track and cancel tasks on shutdown
struct Worker {
    task: JoinHandle<()>,
    cancel: CancellationToken,
}

impl Worker {
    fn new() -> Self {
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = token.cancelled() => break,
                    _ = do_work() => {}
                }
            }
        });
        Self { task, cancel }
    }
    
    async fn shutdown(self) {
        self.cancel.cancel();
        let _ = self.task.await;
    }
}
```

---

## Error Handling Anti-Patterns

### 1. Unwrap in Production Code

```rust
// ❌ CRITICAL: Panics in production
let config = load_config().unwrap();
let value = map.get("key").unwrap();

// ✅ GOOD: Proper error handling
let config = load_config().context("Failed to load config")?;
let value = map.get("key").ok_or_else(|| anyhow!("Key not found"))?;

// ✅ ACCEPTABLE: Only when PROVEN impossible to fail
let regex = Regex::new(r"^\d+$").expect("Hardcoded regex is valid");
```

### 2. Swallowing Errors

```rust
// ❌ BAD: Error silently ignored
let _ = send_notification(user);

// ❌ BAD: Log but don't handle
if let Err(e) = send_notification(user) {
    log::error!("Failed: {}", e);
    // Then what?
}

// ✅ GOOD: Handle or propagate
send_notification(user)
    .context("Failed to notify user")?;

// ✅ GOOD: Explicit ignore with reason
let _ = send_notification(user); // Best-effort, user already informed via email
```

### 3. Stringly-Typed Errors

```rust
// ❌ BAD: Can't match on error types
fn process() -> Result<(), String> {
    Err("Something went wrong".to_string())
}

// ✅ GOOD: Typed errors
#[derive(Error, Debug)]
enum ProcessError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Connection failed")]
    ConnectionFailed(#[from] ConnectionError),
}
```

---

## Database Pitfalls

### 1. N+1 Query Problem

```rust
// ❌ BAD: N+1 queries
async fn get_users_with_orders(user_ids: &[Uuid]) -> Vec<UserWithOrders> {
    let mut result = vec![];
    for id in user_ids {
        let user = get_user(id).await?;
        let orders = get_orders_for_user(id).await?; // N queries!
        result.push(UserWithOrders { user, orders });
    }
    result
}

// ✅ GOOD: Single query with JOIN atau 2 batch queries
async fn get_users_with_orders(user_ids: &[Uuid]) -> Vec<UserWithOrders> {
    let users = get_users_batch(user_ids).await?;
    let orders = get_orders_batch(user_ids).await?;
    
    // Combine in memory
    users.into_iter().map(|u| {
        let user_orders = orders.iter()
            .filter(|o| o.user_id == u.id)
            .cloned()
            .collect();
        UserWithOrders { user: u, orders: user_orders }
    }).collect()
}
```

### 2. Missing Transaction

```rust
// ❌ BAD: Inconsistent state if second query fails
async fn transfer_funds(from: Uuid, to: Uuid, amount: i64) -> Result<()> {
    debit_account(from, amount).await?;
    credit_account(to, amount).await?; // If this fails, money is lost!
    Ok(())
}

// ✅ GOOD: Wrap in transaction
async fn transfer_funds(pool: &PgPool, from: Uuid, to: Uuid, amount: i64) -> Result<()> {
    let mut tx = pool.begin().await?;
    
    sqlx::query!("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, from)
        .execute(&mut *tx)
        .await?;
    
    sqlx::query!("UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, to)
        .execute(&mut *tx)
        .await?;
    
    tx.commit().await?;
    Ok(())
}
```

### 3. Connection Pool Starvation

```rust
// ❌ BAD: Long-running transaction holds connection
async fn process_all(pool: &PgPool) -> Result<()> {
    let mut tx = pool.begin().await?;
    
    for item in get_all_items().await? {
        process_item(&mut tx, item).await?;
        external_api_call().await?; // Holds connection while waiting!
    }
    
    tx.commit().await?;
    Ok(())
}

// ✅ GOOD: Batch with smaller transactions
async fn process_all(pool: &PgPool) -> Result<()> {
    let items = get_all_items().await?;
    
    for chunk in items.chunks(100) {
        let mut tx = pool.begin().await?;
        for item in chunk {
            process_item(&mut tx, item).await?;
        }
        tx.commit().await?;
    }
    Ok(())
}
```

---

## Concurrency Issues

### 1. Race Condition in Check-Then-Act

```rust
// ❌ BAD: TOCTOU (Time-Of-Check to Time-Of-Use)
async fn create_user_if_not_exists(email: &str) -> Result<User> {
    if let Some(user) = find_by_email(email).await? {
        return Ok(user);
    }
    // Another request might create user here!
    create_user(email).await
}

// ✅ GOOD: Use database constraints + upsert
async fn create_user_if_not_exists(email: &str) -> Result<User> {
    sqlx::query_as!(User,
        r#"
        INSERT INTO users (email) VALUES ($1)
        ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
        RETURNING *
        "#,
        email
    )
    .fetch_one(&pool)
    .await
}
```

### 2. Shared Mutable State Without Synchronization

```rust
// ❌ BAD: Data race (won't compile in safe Rust, but pattern is wrong)
static mut COUNTER: u64 = 0; // Don't do this!

// ✅ GOOD: Use atomic types
use std::sync::atomic::{AtomicU64, Ordering};
static COUNTER: AtomicU64 = AtomicU64::new(0);

fn increment() -> u64 {
    COUNTER.fetch_add(1, Ordering::SeqCst)
}

// ✅ GOOD: Use proper synchronization
use parking_lot::Mutex;
static CACHE: Mutex<HashMap<String, Value>> = Mutex::new(HashMap::new());
```

---

## Security Vulnerabilities

### 1. SQL Injection

```rust
// ❌ CRITICAL: SQL injection vulnerability
async fn find_user(name: &str) -> Result<User> {
    let query = format!("SELECT * FROM users WHERE name = '{}'", name);
    sqlx::query_as(&query).fetch_one(&pool).await
}

// ✅ GOOD: Parameterized queries
async fn find_user(name: &str) -> Result<User> {
    sqlx::query_as!(User, "SELECT * FROM users WHERE name = $1", name)
        .fetch_one(&pool)
        .await
}
```

### 2. Path Traversal

```rust
// ❌ CRITICAL: Path traversal
async fn serve_file(filename: &str) -> Result<Vec<u8>> {
    let path = format!("/uploads/{}", filename);
    tokio::fs::read(&path).await.map_err(Into::into)
}
// Attacker: filename = "../../../etc/passwd"

// ✅ GOOD: Validate and canonicalize
async fn serve_file(filename: &str) -> Result<Vec<u8>> {
    let base = Path::new("/uploads").canonicalize()?;
    let requested = base.join(filename).canonicalize()?;
    
    if !requested.starts_with(&base) {
        return Err(anyhow!("Invalid path"));
    }
    
    tokio::fs::read(&requested).await.map_err(Into::into)
}
```

### 3. Timing Attacks on Password Comparison

```rust
// ❌ BAD: Timing attack vulnerable
fn verify_token(provided: &str, expected: &str) -> bool {
    provided == expected // Early return leaks info
}

// ✅ GOOD: Constant-time comparison
use subtle::ConstantTimeEq;

fn verify_token(provided: &[u8], expected: &[u8]) -> bool {
    provided.ct_eq(expected).into()
}
```

---

## Performance Killers

### 1. Logging in Hot Paths

```rust
// ❌ BAD: Format string even when not logging
fn process_item(item: &Item) {
    debug!("Processing item: {:?}", item); // Always formats!
}

// ✅ GOOD: Use lazy formatting
fn process_item(item: &Item) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        debug!(?item, "Processing item");
    }
}
```

### 2. Unnecessary Allocations in Loops

```rust
// ❌ BAD: Allocates Vec every iteration
for _ in 0..1000 {
    let buffer: Vec<u8> = Vec::with_capacity(1024);
    process(&buffer);
}

// ✅ GOOD: Reuse allocation
let mut buffer: Vec<u8> = Vec::with_capacity(1024);
for _ in 0..1000 {
    buffer.clear();
    process(&buffer);
}
```

### 3. Box<dyn Error> in Hot Paths

```rust
// ❌ BAD: Box allocation on every error
fn parse(s: &str) -> Result<i32, Box<dyn Error>> {
    s.parse().map_err(|e| Box::new(e) as Box<dyn Error>)
}

// ✅ GOOD: Concrete error type
#[derive(Error, Debug)]
#[error("Parse error: {0}")]
struct ParseError(#[from] std::num::ParseIntError);

fn parse(s: &str) -> Result<i32, ParseError> {
    s.parse().map_err(Into::into)
}
```

---

## Pre-Commit Checklist

Sebelum commit, pastikan:

```bash
# Clippy dengan semua warnings
cargo clippy -- -D warnings -W clippy::pedantic

# Format check
cargo fmt --check

# Test dengan output
cargo test

# Security audit
cargo audit

# Unused dependencies
cargo +nightly udeps
```
