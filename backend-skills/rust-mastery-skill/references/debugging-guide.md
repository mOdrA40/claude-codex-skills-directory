# Debugging Guide

Panduan debugging komprehensif untuk senior Rust developer.

## Table of Contents

1. [Mental Model](#mental-model)
2. [Compiler Errors](#compiler-errors)
3. [Runtime Debugging](#runtime-debugging)
4. [Async Debugging](#async-debugging)
5. [Performance Debugging](#performance-debugging)
6. [Production Debugging](#production-debugging)
7. [Tools](#tools)

---

## Mental Model

### The Senior Engineer Debugging Process

```
1. REPRODUCE → Konsisten reproduce issue
2. ISOLATE   → Narrow down scope
3. UNDERSTAND → Understand root cause (bukan symptoms)
4. FIX       → Fix root cause, bukan workaround
5. VERIFY    → Test fix menyeluruh
6. PREVENT   → Add test, improve code to prevent recurrence
```

### Rule #1: Trust the Compiler

```rust
// Rust compiler adalah debugging partner terbaik Anda
// Jika compiler komplain, 99% compiler benar

// Baca error message dari BAWAH ke ATAS
// Error pertama sering menyebabkan cascade errors
```

---

## Compiler Errors

### Borrow Checker Errors

**Error: "cannot borrow as mutable because it is also borrowed as immutable"**

```rust
// ❌ Problem
fn problematic() {
    let mut data = vec![1, 2, 3];
    let first = &data[0];      // Immutable borrow
    data.push(4);              // Mutable borrow - ERROR!
    println!("{}", first);
}

// ✅ Solution 1: Clone if cheap
fn solution1() {
    let mut data = vec![1, 2, 3];
    let first = data[0];       // Copy, bukan borrow
    data.push(4);
    println!("{}", first);
}

// ✅ Solution 2: Limit borrow scope
fn solution2() {
    let mut data = vec![1, 2, 3];
    {
        let first = &data[0];
        println!("{}", first);
    }  // Borrow ends here
    data.push(4);
}

// ✅ Solution 3: Index instead of reference
fn solution3() {
    let mut data = vec![1, 2, 3];
    let first_idx = 0;
    data.push(4);
    println!("{}", data[first_idx]);
}
```

**Error: "value moved here" / "use of moved value"**

```rust
// ❌ Problem
fn problematic() {
    let s = String::from("hello");
    process(s);                    // s moved here
    println!("{}", s);             // ERROR: s sudah moved
}

// ✅ Solution 1: Clone
fn solution1() {
    let s = String::from("hello");
    process(s.clone());
    println!("{}", s);
}

// ✅ Solution 2: Borrow instead of move
fn solution2() {
    let s = String::from("hello");
    process_ref(&s);               // Pass reference
    println!("{}", s);
}

// ✅ Solution 3: Return ownership
fn solution3() {
    let s = String::from("hello");
    let s = process_and_return(s);
    println!("{}", s);
}
```

### Lifetime Errors

**Error: "missing lifetime specifier"**

```rust
// ❌ Problem
struct Config {
    name: &str,  // Missing lifetime
}

// ✅ Solution 1: Add lifetime
struct Config<'a> {
    name: &'a str,
}

// ✅ Solution 2: Own the data (sering lebih simple)
struct Config {
    name: String,
}

// ✅ Solution 3: Use Cow for flexibility
use std::borrow::Cow;

struct Config<'a> {
    name: Cow<'a, str>,
}
```

**Error: "lifetime may not live long enough"**

```rust
// ❌ Problem
fn longest(a: &str, b: &str) -> &str {
    if a.len() > b.len() { a } else { b }
}

// ✅ Solution: Explicit lifetime annotation
fn longest<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() > b.len() { a } else { b }
}
```

---

## Runtime Debugging

### Using dbg!() Macro

```rust
// dbg!() prints file, line, expression, and value
fn calculate(x: i32, y: i32) -> i32 {
    let sum = dbg!(x + y);          // [src/main.rs:3] x + y = 15
    let result = dbg!(sum * 2);     // [src/main.rs:4] sum * 2 = 30
    result
}

// Works with expressions
let value = dbg!(some_complex_function());

// Chain with Option/Result
let result = some_option
    .map(|x| dbg!(x * 2))
    .filter(|&x| dbg!(x > 10));
```

### Structured Logging dengan tracing

```rust
use tracing::{debug, info, warn, error, instrument, span, Level};

// Function-level tracing
#[instrument(skip(pool), fields(user_id = %user_id))]
async fn get_user(pool: &PgPool, user_id: Uuid) -> Result<User> {
    debug!("Starting user lookup");
    
    let user = sqlx::query_as!(User, "SELECT * FROM users WHERE id = $1", user_id)
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
            Err(AppError::NotFound)
        }
    }
}

// Span untuk logical groupings
async fn process_order(order: Order) {
    let span = span!(Level::INFO, "process_order", order_id = %order.id);
    let _enter = span.enter();
    
    // All logs in this scope will include order_id
    validate_order(&order).await;
    charge_payment(&order).await;
    send_confirmation(&order).await;
}
```

### panic! Debugging

```rust
// Enable full backtraces
// Bash: RUST_BACKTRACE=1 cargo run
// Bash: RUST_BACKTRACE=full cargo run (lebih detail)

// Custom panic hook
fn main() {
    std::panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();
        
        eprintln!("Panic occurred!");
        eprintln!("Info: {}", panic_info);
        eprintln!("Backtrace:\n{}", backtrace);
        
        // Log to monitoring system
        error!("PANIC: {} \n{}", panic_info, backtrace);
    }));
}
```

---

## Async Debugging

### tokio-console

Real-time async task debugger.

```toml
# Cargo.toml
[dependencies]
console-subscriber = "0.2"
tokio = { version = "1", features = ["full", "tracing"] }
```

```rust
// main.rs
fn main() {
    console_subscriber::init();  // Enable tokio-console
    
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            // Your app
        });
}
```

```bash
# Run tokio-console
cargo install tokio-console
tokio-console
```

### Debugging Deadlocks

```rust
// Detect potential deadlock with timeout
use tokio::time::{timeout, Duration};

async fn safe_lock(mutex: &Mutex<Data>) -> Result<MutexGuard<'_, Data>> {
    timeout(Duration::from_secs(5), mutex.lock())
        .await
        .map_err(|_| {
            error!("Potential deadlock detected!");
            AppError::Deadlock
        })
}

// Add lock ordering to prevent deadlocks
// Always acquire locks in consistent order: A -> B -> C
```

### Tracing Async Task Spawning

```rust
use tracing::Instrument;

async fn parent_task() {
    let span = tracing::info_span!("parent_task");
    
    tokio::spawn(
        async {
            // This task inherits parent's span context
            child_work().await;
        }
        .instrument(span.clone())
    );
}
```

---

## Performance Debugging

### CPU Profiling dengan flamegraph

```bash
# Install
cargo install flamegraph

# Profile (Linux)
cargo flamegraph --bin myapp

# Profile specific benchmark
cargo flamegraph --bench my_benchmark
```

### Memory Profiling dengan DHAT

```toml
# Cargo.toml
[profile.release]
debug = true  # Enable debug info for profiling
```

```rust
// Heap profiling
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    
    // Your code
}
```

### Criterion Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_function(c: &mut Criterion) {
    let data = setup_data();
    
    c.bench_function("my_function", |b| {
        b.iter(|| {
            my_function(black_box(&data))
        })
    });
    
    // Compare implementations
    let mut group = c.benchmark_group("implementations");
    group.bench_function("impl_a", |b| b.iter(|| impl_a(black_box(&data))));
    group.bench_function("impl_b", |b| b.iter(|| impl_b(black_box(&data))));
    group.finish();
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
```

```bash
# Run benchmarks
cargo bench

# Compare with baseline
cargo bench -- --save-baseline main
git checkout feature-branch
cargo bench -- --baseline main
```

---

## Production Debugging

### Structured Logging for Production

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

fn init_production_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,sqlx=warn"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .json()                         // JSON format
                .with_current_span(true)        // Include span info
                .with_span_list(true)           // Include span hierarchy
                .with_file(true)                // Include file
                .with_line_number(true)         // Include line
        )
        .init();
}
```

### Request Tracing dengan Request ID

```rust
use axum::{
    middleware::{self, Next},
    extract::Request,
    response::Response,
};
use uuid::Uuid;

async fn request_id_middleware(
    mut request: Request,
    next: Next,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    
    request.headers_mut().insert(
        "x-request-id",
        request_id.parse().unwrap()
    );
    
    let span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %request.method(),
        uri = %request.uri(),
    );
    
    async move {
        let response = next.run(request).await;
        tracing::info!(status = %response.status(), "Request completed");
        response
    }
    .instrument(span)
    .await
}
```

### Error Context Chain

```rust
use anyhow::{Context, Result};

async fn process_order(order_id: Uuid) -> Result<()> {
    let order = fetch_order(order_id)
        .await
        .with_context(|| format!("Failed to fetch order {}", order_id))?;
    
    let payment = charge_payment(&order)
        .await
        .with_context(|| format!("Failed to charge payment for order {}", order_id))?;
    
    send_confirmation(&order, &payment)
        .await
        .with_context(|| format!("Failed to send confirmation for order {}", order_id))?;
    
    Ok(())
}

// Error chain in logs:
// Error: Failed to send confirmation for order 123
// Caused by:
//   0: SMTP connection failed
//   1: Connection refused
```

---

## Tools

### Essential CLI Tools

```bash
# Install
cargo install cargo-watch     # Auto-reload
cargo install cargo-expand    # Macro expansion
cargo install cargo-audit     # Security audit
cargo install cargo-outdated  # Check dependencies
cargo install cargo-bloat     # Binary size analysis
cargo install cargo-llvm-cov  # Code coverage
cargo install cargo-nextest   # Better test runner
cargo install tokio-console   # Async debugger
cargo install flamegraph      # Profiling

# Usage
cargo watch -x check -x test -x run     # Watch mode
cargo expand                             # Expand macros
cargo +nightly expand module::function   # Specific function
cargo audit                              # Check vulnerabilities
cargo outdated                           # Check updates
cargo bloat --release                    # Size analysis
cargo llvm-cov                           # Coverage
cargo nextest run                        # Fast tests
```

### IDE Setup (VS Code)

```json
// .vscode/settings.json
{
    "rust-analyzer.check.command": "clippy",
    "rust-analyzer.checkOnSave.allTargets": true,
    "rust-analyzer.inlayHints.chainingHints.enable": true,
    "rust-analyzer.inlayHints.parameterHints.enable": true,
    "rust-analyzer.lens.references.enable": true,
    "rust-analyzer.lens.implementations.enable": true,
    "rust-analyzer.diagnostics.experimental.enable": true
}
```

### LLDB/GDB Debugging

```bash
# Build with debug symbols
cargo build

# Run with LLDB
lldb target/debug/myapp

# Common LLDB commands
(lldb) breakpoint set --name main
(lldb) breakpoint set --file src/lib.rs --line 42
(lldb) run
(lldb) next       # Step over
(lldb) step       # Step into
(lldb) continue
(lldb) print variable_name
(lldb) bt         # Backtrace
```

### rust-gdb/rust-lldb

```bash
# Rust-aware debuggers
rust-gdb target/debug/myapp
rust-lldb target/debug/myapp

# Pretty-printing for Rust types
(lldb) type summary add --summary-string "${var.0}" "alloc::string::String"
```

---

## Debugging Checklist

Ketika stuck:

1. **Baca error message lengkap** - Compiler Rust sangat helpful
2. **Minimal reproduction** - Strip kode sampai bug masih muncul
3. **Check assumptions** - dbg!() setiap langkah
4. **Read the docs** - Library docs sering punya caveats
5. **Search issues** - GitHub issues library terkait
6. **Rubber duck** - Explain problem out loud
7. **Take a break** - Fresh eyes solve bugs faster
