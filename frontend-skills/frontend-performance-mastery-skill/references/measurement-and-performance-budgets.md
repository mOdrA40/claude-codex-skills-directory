# Measurement and Performance Budgets

## Principle

If a team does not define what “fast enough” means, optimization turns into noise and debate.

## Define Budgets For

- initial route load
- interaction responsiveness
- client bundle size
- image/media cost
- memory growth over long sessions

## Budgeting Heuristics

### Tie budgets to user journeys

Budgeting only homepage load is not enough if the real product pain lives in dashboards, editors, or media-heavy flows.

### Split budgets by route class and device class

The same threshold rarely makes sense for every route or every device cohort.

### Define response rules, not just numbers

If a budget is violated, teams should know:

- who investigates
- what signal confirms severity
- what mitigation paths exist

## Failure Modes

- optimizing what is easy to measure instead of what users feel
- no clear threshold for regression
- dashboards full of signals with no decision rule

### Budget theater

Numbers exist in documentation, but no release, alerting, or prioritization behavior changes when they are exceeded.

## Review Questions

- what user pain does this metric represent?
- what threshold is unacceptable?
- who owns response when budgets are violated?
- which route or journey currently lacks a meaningful budget entirely?
