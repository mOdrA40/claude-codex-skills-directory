# Zig Allocators and Memory Strategy for Backends

## Why This Matters

In production backends, memory is not an implementation detail. It directly affects latency, throughput, tail behavior, fragmentation, and failure modes. Zig gives you explicit allocator control, which is powerful only if the team uses it deliberately.

## Allocator Selection Heuristics

Use allocator choice as an architectural decision.

### `page_allocator`

Use when:

- bootstrapping very small processes
- low-frequency setup allocations
- simple command-line tools

Do not use as a casual default for request-path heavy services.

### `GeneralPurposeAllocator`

Use when:

- building general backend logic
- needing debug support during development
- the allocation profile is mixed and still evolving

Tradeoff:

- more flexible
- may cost more than specialized strategies

### `ArenaAllocator`

Use when:

- most allocations share the same lifetime
- request-scoped data can be released together
- parsing and transformation workloads dominate

Tradeoff:

- excellent for request-scoped cleanup
- dangerous if arena lifetime silently expands

### `FixedBufferAllocator`

Use when:

- you need hard caps
- payload size is predictable
- deterministic memory use matters more than flexibility

Tradeoff:

- great for bounded parsing or temporary work
- can fail early under larger-than-expected inputs

## Request-Scoped Strategy

A pragmatic backend default is:

- process-level allocator for long-lived infrastructure
- request-level arena or bounded allocator for parsing and response building
- explicit ownership transfer for data that must outlive the request

## Bad vs Good: Leaking Request Memory

```zig
// ❌ BAD: returning arena-backed memory that outlives the request scope.
pub fn buildResponse(allocator: std.mem.Allocator) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    return try std.fmt.allocPrint(a, "hello", .{});
}
```

```zig
// ✅ GOOD: caller owns the allocator for returned memory.
pub fn buildResponse(allocator: std.mem.Allocator) ![]u8 {
    return try std.fmt.allocPrint(allocator, "hello", .{});
}
```

## Ownership Rules

Document these rules in code review:

- who allocates
- who frees
- whether returned memory is borrowed or owned
- whether slices remain valid after function return
- whether a cache or pool can retain references

## `defer` and `errdefer`

Use `defer` for the normal lifecycle and `errdefer` for partial construction rollback.

```zig
pub fn initService(allocator: std.mem.Allocator) !Service {
    var client = try HttpClient.init(allocator);
    errdefer client.deinit();

    var pool = try DbPool.init(allocator);
    errdefer pool.deinit();

    return Service{
        .client = client,
        .pool = pool,
    };
}
```

## Production Memory Guardrails

- Set hard maximum body sizes.
- Bound per-request temporary memory.
- Avoid unbounded buffering when proxying upstream responses.
- Measure allocation rate in hot paths.
- Treat allocator exhaustion as a first-class failure mode.
- Define how the service behaves under memory pressure.

## Bad vs Good: Unbounded Reads

```zig
// ❌ BAD: trusts the client to send something reasonable.
const body = try reader.readAllAlloc(allocator, std.math.maxInt(usize));
```

```zig
// ✅ GOOD: cap the request body explicitly.
const max_body_bytes = 1 * 1024 * 1024;
const body = try reader.readAllAlloc(allocator, max_body_bytes);
```

## Operational Questions

- What is the maximum memory consumed by one request?
- What inputs create pathological allocation behavior?
- Is cleanup deterministic for failed requests?
- Can backlogged work hold memory longer than intended?
- Which metrics indicate memory pressure before OOM?

## Review Checklist

- No borrowed slices escape owned lifetimes.
- Allocators are passed explicitly where ownership matters.
- Cleanup exists for all error paths.
- Request-size and parse-depth limits are enforced.
- Memory strategy matches workload shape, not aesthetics.
