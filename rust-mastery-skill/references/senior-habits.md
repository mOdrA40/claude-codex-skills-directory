# Senior Engineer Habits & Mindset

Habits dan mindset yang membedakan senior engineer dari yang lain.

## The Senior Engineer Mindset

### 1. Simplicity Over Cleverness

```rust
// ❌ "Clever" code
fn f<T: Into<String>>(x: impl AsRef<str>) -> impl Iterator<Item = char> {
    x.as_ref().chars().filter(|c| c.is_alphabetic())
}

// ✅ Simple, readable code
fn extract_letters(text: &str) -> Vec<char> {
    text.chars().filter(|c| c.is_alphabetic()).collect()
}
```

**Mantra:**
- "Make it work, make it right, make it fast" - dalam urutan itu
- Code yang paling mudah di-maintain adalah code yang tidak ada
- Jika butuh comment untuk menjelaskan, mungkin kode terlalu kompleks

### 2. Think in Tradeoffs

Setiap keputusan teknis adalah tradeoff. Senior engineers:

```
✅ "Opsi A lebih cepat tapi lebih kompleks. Opsi B lebih simple tapi 
   10ms lebih lambat. Untuk use case kita dengan 100 req/s, 
   simplicity lebih penting."

❌ "Opsi A adalah yang terbaik karena lebih optimal."
```

**Common Tradeoffs:**
- Performance vs Readability
- Flexibility vs Simplicity
- Build vs Buy
- Consistency vs Availability
- Time-to-market vs Technical debt

### 3. Own the Problem, Not Just the Code

```
Junior: "Saya sudah push kode, kalau ada bug bukan salah saya."

Senior: "Saya akan pastikan fitur ini bekerja end-to-end, 
        monitor production, dan handle issues yang muncul."
```

**Ownership Spectrum:**
1. Menulis kode ✓
2. Menulis tests ✓
3. Review & merge ✓
4. Deploy ✓
5. Monitor ✓
6. Respond to incidents ✓
7. Improve system based on learnings ✓

### 4. Communicate Proactively

```
❌ Silent failure
   *Bug ditemukan di production 3 hari kemudian*
   "Oh iya, saya tahu itu bisa terjadi."

✅ Proactive communication
   "Hey team, saya menemukan edge case di deployment kemarin. 
   Sudah saya fix di PR #123. Tidak ada data loss, tapi 
   kita perlu tambah monitoring untuk kasus serupa."
```

---

## Daily Habits

### Morning Routine

```markdown
1. Check alerts/monitors (5 min)
   - Apakah ada yang terjadi overnight?
   - Apakah ada tickets urgent?

2. Review PRs waiting on you (15 min)
   - Jangan block teammates

3. Plan the day (5 min)
   - 1 important task
   - 2-3 smaller tasks
```

### Coding Habits

```bash
# Before starting
git pull --rebase
cargo check

# During development
cargo watch -x check  # Auto-check on save

# Before commit
cargo fmt
cargo clippy -- -D warnings
cargo test

# Before PR
git rebase -i main  # Clean up commits
```

### Code Organization

```rust
// Group imports logically
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use sqlx::PgPool;
use tracing::{info, warn};

use crate::domain::User;
use crate::infrastructure::Repository;

// Order functions by visibility, then by logical flow
impl UserService {
    // Public API first
    pub fn new() -> Self { }
    pub async fn create_user() { }
    pub async fn get_user() { }
    
    // Private helpers second
    fn validate_input() { }
    async fn notify_user() { }
}
```

### Documentation Habits

```rust
// Document WHY, not WHAT
// ❌ BAD: "Increments counter by 1"
// ✅ GOOD: "Track failed login attempts for rate limiting"

/// Validates user input before database insertion.
/// 
/// We validate at this layer (not in the API layer) because
/// the same validation applies whether input comes from API,
/// CLI, or batch processing.
fn validate_user(input: &CreateUserInput) -> Result<()> {
    // ...
}
```

---

## Problem-Solving Approach

### The 5 Whys

```
Problem: API response slow

Why 1: Database query takes 2 seconds
Why 2: Query doing full table scan
Why 3: Missing index on user_id column
Why 4: Migration wasn't run in production
Why 5: Deployment process doesn't verify migrations

Root cause: Deployment process gap
Fix: Add migration verification to CI/CD
```

### Debugging Process

```
1. REPRODUCE
   - Can I consistently trigger the bug?
   - What are the exact steps?

2. ISOLATE
   - What changed recently?
   - What's the smallest case that fails?

3. UNDERSTAND
   - What's the actual vs expected behavior?
   - What assumptions might be wrong?

4. FIX
   - Fix root cause, not symptoms
   - Write test first if possible

5. VERIFY
   - Does original case pass?
   - Are there similar bugs elsewhere?

6. PREVENT
   - Why wasn't this caught earlier?
   - How can we prevent similar issues?
```

---

## Technical Decision Making

### RFC Template (untuk keputusan besar)

```markdown
# RFC: [Title]

## Context
Apa problem yang kita solve?

## Proposal
Apa solusi yang diusulkan?

## Alternatives Considered
Opsi lain apa yang dipertimbangkan dan kenapa tidak dipilih?

## Tradeoffs
- Pro: ...
- Con: ...

## Migration Path
Bagaimana kita transition dari state sekarang?

## Timeline
Estimasi effort dan timeline

## Decision
[Akan diisi setelah discussion]
```

### When to Push Back

```
✅ Push back ketika:
- Timeline unrealistic tanpa cutting corners
- Requirements unclear atau contradictory
- Technical approach akan create technical debt
- Security atau reliability compromised

❌ Jangan push back untuk:
- Personal preference tentang tools/framework
- "Not invented here" syndrome
- Resistance to change
```

---

## Working with Others

### Mentoring

```
Senior engineer bukan yang paling banyak tahu,
tapi yang paling banyak membantu tim sukses.
```

**Techniques:**
- Pair programming - show, don't tell
- Ask questions instead of giving answers
- Celebrate small wins
- Give specific, actionable feedback

### Code Review Philosophy

```
✅ "Have you considered using X? It might handle edge case Y better."
✅ "Nice solution! One suggestion: ..."
✅ "I don't understand this part. Can you explain?"

❌ "This is wrong."
❌ "You should have done it like I would."
❌ "Obviously this is the better way."
```

### Handling Disagreements

```
1. Assume good intent
2. Focus on facts, not opinions
3. Ask clarifying questions
4. Propose experiments if needed
5. Disagree and commit when decision is made
```

---

## Career Development

### T-Shaped Skills

```
Broad knowledge across many areas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ┃
       ┃ Deep expertise
       ┃ in 1-2 areas
       ┃
       ┃
       ┃
```

### Continuous Learning

```markdown
Weekly:
- [ ] Read 1 technical article
- [ ] Review interesting open source PR

Monthly:
- [ ] Try 1 new tool/library
- [ ] Write 1 knowledge sharing doc

Quarterly:
- [ ] Deep dive into unfamiliar area
- [ ] Present learning to team
```

### Building Influence

```
Junior: Impact through own code
Mid: Impact through team's code
Senior: Impact through organization's decisions
Staff+: Impact through industry practices
```

---

## Red Flags to Watch For

### In Yourself

- "I can do it faster myself" → Opportunity to mentor
- "This isn't my job" → Ownership gap
- "We've always done it this way" → Resistance to improvement
- "I'll fix it later" → Technical debt accumulation
- "They should know better" → Communication gap

### In Systems

- No tests → Afraid to change
- No monitoring → Flying blind
- No documentation → Tribal knowledge
- No automation → Human error prone
- No review → Quality gap

### In Teams

- Hero culture → Bus factor of 1
- Blame culture → People hide mistakes
- Silos → Knowledge isolation
- Meeting heavy → No time to think
- Always urgent → No time to improve

---

## Final Wisdom

```
"The best code is no code at all."
"Ship, then iterate."
"Optimize for understanding, not performance."
"Every line of code is a liability."
"Simple things should be simple, complex things should be possible."
```
