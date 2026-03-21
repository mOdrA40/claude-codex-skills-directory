# Trace Architecture and Component Roles (Jaeger)

## Rules

- Collectors, query services, agents, and storage backends form one reliability chain.
- Architecture should reflect scale, latency expectations, and operator maturity.
- Simplicity often beats overly customized tracing topologies.
- Component responsibilities should be explicit to operators and developers.

## Design Guidance

- Decide which pieces are still justified in your deployment model.
- Match deployment topology to collector load, backend constraints, and query patterns.
- Keep platform responsibilities visible across teams.
- Document what failure in each component means for trace trustworthiness.

## Principal Review Lens

- Which component is least understood but most critical?
- Are we running unnecessary complexity from historical deployment choices?
- What trace truth is lost if one component fails?
- Which simplification would improve reliability most?
