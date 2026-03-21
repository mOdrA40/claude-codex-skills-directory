# Phoenix and Service Boundaries

## Goal

Phoenix should make transport concerns easy. It should not become the place where business rules, downstream orchestration, retries, and side effects are mixed together.

## Boundary Direction

Preferred flow:

`controller/liveview/channel -> application service -> domain -> adapters`

Keep controllers thin:

- parse and validate input
- authenticate and authorize
- attach request metadata
- call a service
- map errors to HTTP semantics

## Bad vs Good: Fat Controller

```elixir
# ❌ BAD: controller performs validation, DB logic, and external calls inline.
def create(conn, params) do
  changeset = User.changeset(%User{}, params)

  case Repo.insert(changeset) do
    {:ok, user} ->
      HTTPoison.post!("https://billing.example.com/sync", Jason.encode!(%{id: user.id}))
      json(conn, %{id: user.id})

    {:error, changeset} ->
      conn |> put_status(:unprocessable_entity) |> json(%{errors: changeset})
  end
end
```

```elixir
# ✅ GOOD: controller delegates to a boundary.
def create(conn, params) do
  with {:ok, input} <- CreateUserInput.cast(params),
       {:ok, user} <- Accounts.create_user(input, request_context(conn)) do
    conn |> put_status(:created) |> json(%{id: user.id})
  else
    {:error, reason} -> render_error(conn, reason)
  end
end
```

## Context Boundaries

Phoenix contexts are useful when they model a real domain boundary. They become harmful when they turn into giant buckets of unrelated functions.

Prefer contexts that represent:

- Accounts
- Billing
- Fulfillment
- Identity
- Notifications

Avoid contexts that represent technical clutter:

- Helpers
- Common
- Utilities
- ServicesEverything

## Error Mapping

Your HTTP edge should consistently map domain errors such as:

- `:invalid_input`
- `:unauthorized`
- `:forbidden`
- `:not_found`
- `:conflict`
- `:dependency_timeout`
- `:dependency_unavailable`

This consistency matters more than framework cleverness.

## Transaction and Side-Effect Policy

Do not mix local DB transactions and uncoordinated outbound side effects casually.

Prefer:

- transactional write
- outbox/event recording
- async delivery or reliable downstream processing

## Review Checklist

- Controllers are thin.
- Domain errors map consistently.
- Outbound IO is not buried in views/controllers.
- Context boundaries are coherent.
- Side effects have retry/idempotency strategy.
