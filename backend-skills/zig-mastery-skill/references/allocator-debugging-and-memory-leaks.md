# Allocator Debugging and Memory Leaks in Zig Services

## Purpose

In production Zig services, memory incidents are rarely just "a leak". They are usually one of several classes of problems:

- ownership confusion
- unbounded request or queue growth
- allocator misuse across boundaries
- failure paths that skip cleanup
- caches or buffers with no practical cap

This guide helps distinguish those cases so debugging leads to the real fix.

## Start With Classification

When memory grows, first ask which kind of problem you are seeing:

- **True leak**
  Memory is allocated and never released.
- **Retained growth**
  Memory is still referenced intentionally, but policy is wrong or unbounded.
- **Burst pressure**
  Peak concurrency or payload size temporarily raises memory use.
- **Fragmentation or allocator behavior**
  Memory patterns create poor reuse or misleading process-level growth.
- **Shutdown / error-path leak**
  The happy path frees correctly, but timeout and failure paths do not.

Do not call everything a leak. The fix depends on the class.

## Ownership Questions

For any suspicious buffer or object, answer:

- who allocates it?
- who owns it after return?
- who frees it?
- what happens when parsing fails halfway?
- what happens when cancellation or timeout interrupts the workflow?

If ownership must be inferred from call-site folklore, the code is already too risky.

## Common Failure Patterns

### 1. Arena Used Beyond Intended Lifetime

Arenas are useful for request-scoped work, but dangerous when data escapes the request boundary.

```zig
// ❌ BAD: returning data tied to a request-scoped arena.
pub fn loadUser() !User {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    return try repo.fetchUser(allocator);
}
```

```zig
// ✅ GOOD: keep lifetime aligned with ownership expectations.
pub fn loadUser(allocator: std.mem.Allocator) !User {
    return try repo.fetchUser(allocator);
}
```

### 2. Cleanup Missing on Error Path

Use `errdefer` when partial allocation or initialization can fail midway.

```zig
var buffer = try allocator.alloc(u8, size);
errdefer allocator.free(buffer);
```

If only the success path cleans up correctly, incidents will appear random under retries and failures.

### 3. Unbounded Read or Queue Accumulation

The bug may not be allocator misuse at all. It may be a workload policy bug:

- request body has no hard limit
- queue depth has no cap
- worker throughput is lower than ingress for long periods
- retries amplify backlog

That is a capacity control issue, not only a coding issue.

## Debugging Workflow

1. Reproduce with representative payload sizes and concurrency.
2. Confirm whether growth is steady, bursty, or tied to specific endpoints/jobs.
3. Inspect hot allocation paths and long-lived owners.
4. Trace failure and cancellation paths with `errdefer` and shutdown handling in mind.
5. Check whether queues, caches, or pooled objects are bounded.
6. Validate whether the fix reduces both average and worst-case memory behavior.

## Signals to Add Before Incidents

At minimum, expose:

- current in-flight requests
- queue depth and oldest job age
- request payload size distribution
- failure rate by operation
- allocator-heavy operation latency
- memory usage or process RSS trend by release version

If you cannot correlate memory growth with one operation, one dependency, or one release, incident response will be slow.

## Review Checklist

- Is allocator ownership explicit at every boundary that returns owned data?
- Do failure paths free partially initialized resources?
- Are request sizes, batch sizes, and queue depth bounded?
- Can one tenant or one endpoint monopolize memory?
- Are caches capped and evicted intentionally?
- Does shutdown release workers, buffers, and client resources predictably?

## Principal Heuristics

- Prefer simple allocator policy that the whole team can reason about.
- Treat memory incidents as architecture signals, not only local code defects.
- If the system needs heroic allocator reasoning to stay safe, simplify the design.
