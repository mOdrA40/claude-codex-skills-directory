# TanStack Form Guide

## Table of Contents
1. [Basic Setup](#basic-setup)
2. [Field Validation](#field-validation)
3. [Async Validation](#async-validation)
4. [Form Submission](#form-submission)
5. [Field Arrays](#field-arrays)
6. [Integration with TanStack Query](#integration-with-query)
7. [Custom Components](#custom-components)

## Basic Setup

```typescript
import { useForm } from '@tanstack/react-form'
import { zodValidator } from '@tanstack/zod-form-adapter'
import { z } from 'zod'

const userSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters'),
  email: z.string().email('Invalid email address'),
  age: z.number().min(18, 'Must be at least 18'),
})

type UserFormData = z.infer<typeof userSchema>

function UserForm() {
  const form = useForm({
    defaultValues: {
      name: '',
      email: '',
      age: 0,
    } satisfies UserFormData,
    onSubmit: async ({ value }) => {
      // Submit logic
      console.log(value)
    },
    validatorAdapter: zodValidator(),
  })

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        e.stopPropagation()
        form.handleSubmit()
      }}
    >
      <form.Field
        name="name"
        validators={{
          onChange: z.string().min(2),
        }}
      >
        {(field) => (
          <div>
            <label htmlFor={field.name}>Name</label>
            <input
              id={field.name}
              value={field.state.value}
              onBlur={field.handleBlur}
              onChange={(e) => field.handleChange(e.target.value)}
            />
            {field.state.meta.errors.length > 0 && (
              <span className="error">{field.state.meta.errors[0]}</span>
            )}
          </div>
        )}
      </form.Field>

      <form.Field name="email">
        {(field) => (
          <div>
            <label htmlFor={field.name}>Email</label>
            <input
              id={field.name}
              type="email"
              value={field.state.value}
              onBlur={field.handleBlur}
              onChange={(e) => field.handleChange(e.target.value)}
            />
            <FieldError field={field} />
          </div>
        )}
      </form.Field>

      <form.Field name="age">
        {(field) => (
          <div>
            <label htmlFor={field.name}>Age</label>
            <input
              id={field.name}
              type="number"
              value={field.state.value}
              onBlur={field.handleBlur}
              onChange={(e) => field.handleChange(Number(e.target.value))}
            />
            <FieldError field={field} />
          </div>
        )}
      </form.Field>

      <form.Subscribe
        selector={(state) => [state.canSubmit, state.isSubmitting]}
      >
        {([canSubmit, isSubmitting]) => (
          <button type="submit" disabled={!canSubmit || isSubmitting}>
            {isSubmitting ? 'Submitting...' : 'Submit'}
          </button>
        )}
      </form.Subscribe>
    </form>
  )
}

// Reusable error component
function FieldError({ field }: { field: FieldApi<any, any, any, any> }) {
  return field.state.meta.isTouched && field.state.meta.errors.length > 0 ? (
    <span className="text-red-500 text-sm">
      {field.state.meta.errors[0]}
    </span>
  ) : null
}
```

## Field Validation

```typescript
import { z } from 'zod'

function ValidatedForm() {
  const form = useForm({
    defaultValues: {
      username: '',
      password: '',
      confirmPassword: '',
    },
    validatorAdapter: zodValidator(),
    // Form-level validation
    validators: {
      onChange: z.object({
        username: z.string(),
        password: z.string(),
        confirmPassword: z.string(),
      }).refine(
        (data) => data.password === data.confirmPassword,
        {
          message: "Passwords don't match",
          path: ['confirmPassword'],
        }
      ),
    },
    onSubmit: async ({ value }) => {
      // Submit
    },
  })

  return (
    <form onSubmit={(e) => { e.preventDefault(); form.handleSubmit() }}>
      <form.Field
        name="username"
        validators={{
          // Validate on change
          onChange: z.string()
            .min(3, 'Username must be at least 3 characters')
            .max(20, 'Username must be at most 20 characters')
            .regex(/^[a-z0-9_]+$/, 'Only lowercase letters, numbers, and underscores'),
          // Validate on blur (more expensive checks)
          onBlur: z.string().min(3),
        }}
      >
        {(field) => (
          <div>
            <input
              value={field.state.value}
              onChange={(e) => field.handleChange(e.target.value)}
              onBlur={field.handleBlur}
            />
            <FieldError field={field} />
          </div>
        )}
      </form.Field>

      <form.Field
        name="password"
        validators={{
          onChange: z.string()
            .min(8, 'Password must be at least 8 characters')
            .regex(/[A-Z]/, 'Must contain uppercase')
            .regex(/[0-9]/, 'Must contain number'),
        }}
      >
        {(field) => (
          <div>
            <input
              type="password"
              value={field.state.value}
              onChange={(e) => field.handleChange(e.target.value)}
              onBlur={field.handleBlur}
            />
            <FieldError field={field} />
          </div>
        )}
      </form.Field>
    </form>
  )
}
```

## Async Validation

```typescript
function AsyncValidatedForm() {
  const form = useForm({
    defaultValues: { username: '', email: '' },
    validatorAdapter: zodValidator(),
    onSubmit: async ({ value }) => {
      // Submit
    },
  })

  return (
    <form onSubmit={(e) => { e.preventDefault(); form.handleSubmit() }}>
      <form.Field
        name="username"
        validators={{
          // Sync validation first
          onChange: z.string().min(3),
          // Async validation on blur (expensive API call)
          onBlurAsync: async ({ value }) => {
            // Debounce handled internally
            const isAvailable = await checkUsernameAvailable(value)
            if (!isAvailable) {
              return 'Username is already taken'
            }
            return undefined
          },
          // Async validation debounce
          onBlurAsyncDebounceMs: 500,
        }}
      >
        {(field) => (
          <div>
            <input
              value={field.state.value}
              onChange={(e) => field.handleChange(e.target.value)}
              onBlur={field.handleBlur}
            />
            {field.state.meta.isValidating && <span>Checking...</span>}
            <FieldError field={field} />
          </div>
        )}
      </form.Field>

      <form.Field
        name="email"
        validators={{
          onChange: z.string().email(),
          // Async validation dengan query
          onBlurAsync: async ({ value }) => {
            const exists = await api.checkEmailExists(value)
            return exists ? 'Email already registered' : undefined
          },
          onBlurAsyncDebounceMs: 300,
        }}
      >
        {(field) => (
          <div>
            <input
              type="email"
              value={field.state.value}
              onChange={(e) => field.handleChange(e.target.value)}
              onBlur={field.handleBlur}
            />
            {field.state.meta.isValidating && <span>Validating...</span>}
            <FieldError field={field} />
          </div>
        )}
      </form.Field>
    </form>
  )
}
```

## Form Submission

```typescript
function SubmissionForm() {
  const form = useForm({
    defaultValues: { name: '', email: '' },
    onSubmit: async ({ value, formApi }) => {
      try {
        await api.createUser(value)
        // Reset form after success
        formApi.reset()
        toast.success('User created!')
      } catch (error) {
        // Set form-level error
        formApi.setErrorMap({
          onSubmit: error.message,
        })
        // Or set field-level error from server
        if (error.field === 'email') {
          formApi.setFieldMeta('email', (prev) => ({
            ...prev,
            errors: [error.message],
          }))
        }
      }
    },
  })

  return (
    <form onSubmit={(e) => { e.preventDefault(); form.handleSubmit() }}>
      {/* Fields... */}

      {/* Form-level error */}
      <form.Subscribe selector={(state) => state.errorMap.onSubmit}>
        {(error) => error && <div className="error">{error}</div>}
      </form.Subscribe>

      {/* Submit button with states */}
      <form.Subscribe
        selector={(state) => ({
          canSubmit: state.canSubmit,
          isSubmitting: state.isSubmitting,
          isValid: state.isValid,
          isDirty: state.isDirty,
        })}
      >
        {({ canSubmit, isSubmitting, isDirty }) => (
          <div>
            <button
              type="submit"
              disabled={!canSubmit || isSubmitting}
            >
              {isSubmitting ? 'Saving...' : 'Save'}
            </button>
            <button
              type="button"
              onClick={() => form.reset()}
              disabled={!isDirty}
            >
              Reset
            </button>
          </div>
        )}
      </form.Subscribe>
    </form>
  )
}
```

## Field Arrays

```typescript
function DynamicForm() {
  const form = useForm({
    defaultValues: {
      users: [{ name: '', email: '' }],
    },
    onSubmit: async ({ value }) => {
      console.log(value.users)
    },
  })

  return (
    <form onSubmit={(e) => { e.preventDefault(); form.handleSubmit() }}>
      <form.Field name="users" mode="array">
        {(field) => (
          <div>
            {field.state.value.map((_, index) => (
              <div key={index} className="user-row">
                <form.Field name={`users[${index}].name`}>
                  {(subField) => (
                    <input
                      placeholder="Name"
                      value={subField.state.value}
                      onChange={(e) => subField.handleChange(e.target.value)}
                    />
                  )}
                </form.Field>

                <form.Field name={`users[${index}].email`}>
                  {(subField) => (
                    <input
                      placeholder="Email"
                      type="email"
                      value={subField.state.value}
                      onChange={(e) => subField.handleChange(e.target.value)}
                    />
                  )}
                </form.Field>

                <button
                  type="button"
                  onClick={() => field.removeValue(index)}
                  disabled={field.state.value.length <= 1}
                >
                  Remove
                </button>
              </div>
            ))}

            <button
              type="button"
              onClick={() => field.pushValue({ name: '', email: '' })}
            >
              Add User
            </button>
          </div>
        )}
      </form.Field>

      <button type="submit">Submit All</button>
    </form>
  )
}
```

## Integration with Query

```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useForm } from '@tanstack/react-form'

function UserEditForm({ userId }: { userId: string }) {
  const queryClient = useQueryClient()
  const { data: user } = useSuspenseQuery(userQueries.detail(userId))

  const updateMutation = useMutation({
    mutationFn: (data: UpdateUserDto) => api.updateUser(userId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.detail(userId) })
      toast.success('User updated!')
    },
  })

  const form = useForm({
    defaultValues: {
      name: user.name,
      email: user.email,
      bio: user.bio ?? '',
    },
    onSubmit: async ({ value }) => {
      await updateMutation.mutateAsync(value)
    },
  })

  return (
    <form onSubmit={(e) => { e.preventDefault(); form.handleSubmit() }}>
      {/* Fields */}

      <form.Subscribe selector={(state) => state.isSubmitting}>
        {(isSubmitting) => (
          <button type="submit" disabled={isSubmitting || updateMutation.isPending}>
            {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
          </button>
        )}
      </form.Subscribe>

      {updateMutation.isError && (
        <div className="error">{updateMutation.error.message}</div>
      )}
    </form>
  )
}
```

## Custom Components

```typescript
// shared/components/form/TextField.tsx
import { useFieldContext } from '@tanstack/react-form'

interface TextFieldProps {
  label: string
  type?: 'text' | 'email' | 'password'
  placeholder?: string
}

export function TextField({ label, type = 'text', placeholder }: TextFieldProps) {
  const field = useFieldContext()

  return (
    <div className="form-field">
      <label htmlFor={field.name} className="form-label">
        {label}
      </label>
      <input
        id={field.name}
        type={type}
        placeholder={placeholder}
        value={field.state.value}
        onChange={(e) => field.handleChange(e.target.value)}
        onBlur={field.handleBlur}
        className={`form-input ${
          field.state.meta.errors.length > 0 ? 'error' : ''
        }`}
      />
      {field.state.meta.isTouched && field.state.meta.errors.length > 0 && (
        <span className="form-error">{field.state.meta.errors[0]}</span>
      )}
    </div>
  )
}

// Usage
<form.Field name="email">
  {() => <TextField label="Email" type="email" />}
</form.Field>
```
