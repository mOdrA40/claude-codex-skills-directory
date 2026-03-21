# Flutter Performance and Rendering

## Focus Areas

- rebuild frequency
- frame budget discipline
- image and list performance
- startup cost
- jank investigation order

## Investigation Order

1. identify whether the issue is startup, transition, list scrolling, animation, or background work
2. isolate the dominant route, widget tree, or device class
3. determine whether the issue is caused by rebuild churn, expensive layout/paint, image handling, or blocking async work
4. compare debug assumptions against real-device release behavior

## Common Failure Modes

### Rebuild storms

Broad state changes cause large widget subtrees to rebuild when only a small region actually changed.

### Smooth in emulator, janky in real devices

Teams validate on comfortable hardware and miss memory, image, or frame-budget issues on actual target devices.

### Startup bloat

The app performs too many initializations before showing meaningful UI, making first-use experience feel slow.

## Review Questions

- which screen or interaction misses frame budget most often?
- what work can be deferred without harming correctness?
- are list, image, or animation costs dominating the user experience?
