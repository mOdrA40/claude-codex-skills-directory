# Code Review Guidelines

Panduan code review dari perspektif senior engineer dengan 20+ tahun pengalaman.

## The Senior Engineer Code Review Mindset

```
"Code review bukan tentang menemukan kesalahan,
tapi tentang meningkatkan kualitas tim secara kolektif."
```

### Review Priorities (Ordered)

1. **Correctness** - Apakah kode benar?
2. **Security** - Apakah ada vulnerability?
3. **Performance** - Apakah ada obvious bottlenecks?
4. **Maintainability** - Apakah kode mudah dipahami dan diubah?
5. **Style** - Consistency dengan codebase

---

## The Review Checklist

### 1. Architecture & Design

```markdown
□ Single Responsibility - Setiap module/function punya satu purpose
□ Appropriate Abstraction - Tidak over-engineered, tidak under-abstracted
□ Dependency Direction - Dependencies mengalir ke arah yang benar
□ Error Boundaries - Error handling di layer yang tepat
□ Testability - Kode bisa di-test dengan mudah
```

### 2. Rust-Specific Checks

```markdown
□ Ownership - Apakah ownership model masuk akal?
□ Lifetimes - Apakah lifetimes necessary dan correct?
□ Error Handling - Result/Option handling proper?
□ Unwrap/Expect - Tidak ada unwrap() di production paths
□ Clone Abuse - Tidak clone tanpa alasan jelas
□ Concurrency - Safe concurrent access patterns
```

**Red Flags:**

```rust
// ❌ Unwrap in production code
let user = users.get(id).unwrap();

// ❌ Silent error handling
let _ = send_email(user);

// ❌ Unnecessary clone
let name = user.name.clone();
process(&name);
```

### 3. Security Checks

```markdown
□ Input validation at boundaries
□ SQL injection impossible (parameterized queries)
□ Authentication/authorization checked
□ Secrets not logged
□ Path traversal prevented
```

---

## How to Give Feedback

### Use Conventional Comments

```
nitpick: Consider renaming this variable for clarity

suggestion: Using `filter_map` would be more idiomatic here

question: What happens if this fails during peak hours?

issue: This could cause a deadlock under concurrent access

praise: Great use of the type-state pattern here!
```

### Be Specific and Actionable

```
// ❌ BAD
"This code is confusing"

// ✅ GOOD
"Consider extracting this logic into a separate function named 
`calculate_discount`. The current function is doing too many things."
```

### Distinguish Must-Fix vs Nice-to-Have

```
// MUST FIX (blocker)
🔴 "This allows SQL injection. Must use parameterized queries."

// SHOULD FIX (important)
🟡 "This N+1 query will cause performance issues at scale."

// NICE TO HAVE (suggestion)
🟢 "nitpick: Consider using `if let` instead of `match` here."
```

---

## Senior Engineer Habits

### 1. Review Your Own Code First

Sebelum request review:
- Run `cargo clippy -- -D warnings`
- Run `cargo fmt`
- Run tests
- Self-review diff
- Write good PR description

### 2. Keep PRs Small

```
< 200 lines: Easy to review ✅
200-400 lines: Acceptable ⚠️
> 400 lines: Consider splitting ❌
```

### 3. Respond Gracefully

```
// Receiving feedback
"Thanks for catching this! Fixed in latest commit."
"Good point. I chose this approach because X, but happy to change."
"I disagree because X, but let's discuss offline if needed."

// NOT
"That's wrong"
"I know what I'm doing"
```

### 4. Review in Layers

```
1st pass: Architecture & overall design (10 min)
2nd pass: Logic correctness & edge cases (15 min)
3rd pass: Code style & minor improvements (5 min)
```

### 5. Don't Be a Gatekeeper

```
✅ Approve with minor comments that can be fixed later
✅ Trust junior devs to make judgment calls
✅ Focus on teaching, not blocking

❌ Hold PR hostage for style preferences
❌ Require every suggestion to be implemented
❌ Review should not take longer than writing the code
```

---

## Code Review Anti-Patterns

### The Nitpicker
Fokus berlebihan pada style daripada substance.

### The Rubber Stamper
Approve tanpa actually reading.

### The Perfectionist
Tidak pernah approve, selalu ada yang kurang.

### The Silent Approver
Approve tanpa feedback, tidak membantu growth.

### The Blocker
Menggunakan review sebagai power play.

---

## Quick Reference Commands

```bash
# Before PR
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo audit

# During Review
git diff main...feature-branch
cargo doc --open  # Check doc generation
```
