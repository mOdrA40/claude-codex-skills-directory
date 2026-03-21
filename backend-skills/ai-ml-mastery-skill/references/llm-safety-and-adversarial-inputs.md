# LLM Safety and Adversarial Inputs in Backend Systems

## Purpose

LLM systems in production are exposed to prompt injection, untrusted context, schema drift, and unsafe tool invocation. This is a backend security and reliability problem, not just a model quality issue.

## Rules

- treat prompts, retrieved documents, and user input as untrusted inputs
- separate system policy from dynamic content clearly
- validate tool inputs and tool outputs
- define refusal and fallback behavior explicitly
- log enough for incident response without storing sensitive raw content casually

## Failure Modes

- prompt injection through retrieved content
- output schema breakage
- unsafe tool invocation
- over-trusting retrieval sources
- hidden token/cost explosion from adversarial prompts

## Review Questions

- what can this model be tricked into doing?
- what tools or data access can the model trigger?
- how is unsafe output contained?
- what signals indicate active abuse or prompt exploitation?
