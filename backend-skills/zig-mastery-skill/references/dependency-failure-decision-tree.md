# Dependency Failure Decision Tree for Zig Services

## Database or Dependency Slow

- protect memory and queue growth first
- reduce optional work
- avoid retries that outlive usefulness
- inspect whether parsing or buffering worsens local pressure

## Cache or Secondary Store Failing

- choose degrade vs fail-fast explicitly
- avoid fallback paths that explode allocation pressure
- protect origin from stampede if applicable

## Queue or Worker Dependency Degraded

- pause toxic workers if replay amplifies backlog
- preserve correctness before draining volume blindly
- protect request-serving resources from background pressure

## Agent Heuristics

- classify dependency as critical, optional, or deferrable
- ask which resource saturates first under failure
- ask what observable signal proves containment is working
