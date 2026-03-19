/**
 * Custom Error Classes
 * 
 * Standardized error handling across the application.
 */

export abstract class AppError extends Error {
  abstract readonly statusCode: number
  abstract readonly code: string
  readonly isOperational = true

  constructor(message: string) {
    super(message)
    Object.setPrototypeOf(this, new.target.prototype)
    Error.captureStackTrace(this, this.constructor)
  }
}

export class BadRequestError extends AppError {
  readonly statusCode = 400
  readonly code = "BAD_REQUEST"

  constructor(message = "Bad request") {
    super(message)
  }
}

export class UnauthorizedError extends AppError {
  readonly statusCode = 401
  readonly code = "UNAUTHORIZED"

  constructor(message = "Authentication required") {
    super(message)
  }
}

export class ForbiddenError extends AppError {
  readonly statusCode = 403
  readonly code = "FORBIDDEN"

  constructor(message = "Access denied") {
    super(message)
  }
}

export class NotFoundError extends AppError {
  readonly statusCode = 404
  readonly code = "NOT_FOUND"

  constructor(resource: string, id?: string) {
    super(id ? `${resource} with id ${id} not found` : `${resource} not found`)
  }
}

export class ConflictError extends AppError {
  readonly statusCode = 409
  readonly code = "CONFLICT"

  constructor(message: string) {
    super(message)
  }
}

export class ValidationError extends AppError {
  readonly statusCode = 400
  readonly code = "VALIDATION_ERROR"
  readonly errors: Record<string, string[]>

  constructor(errors: Record<string, string[]>) {
    super("Validation failed")
    this.errors = errors
  }
}

export class TooManyRequestsError extends AppError {
  readonly statusCode = 429
  readonly code = "TOO_MANY_REQUESTS"

  constructor(message = "Too many requests") {
    super(message)
  }
}

export class InternalError extends AppError {
  readonly statusCode = 500
  readonly code = "INTERNAL_ERROR"

  constructor(message = "Internal server error") {
    super(message)
  }
}
