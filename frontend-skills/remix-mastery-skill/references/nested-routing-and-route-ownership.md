# Nested Routing and Route Ownership in Remix

## Principle

Nested routing is powerful because it mirrors UI composition and data ownership. It becomes messy when teams cannot explain which route owns what responsibility.

## Common Failure Modes

- parent routes becoming universal dumping grounds
- child routes depending on parent assumptions too implicitly
- route boundaries chosen by file convenience instead of product ownership

## Review Questions

- what does this route own?
- what should be inherited vs isolated?
- which route boundary becomes painful during incidents?
