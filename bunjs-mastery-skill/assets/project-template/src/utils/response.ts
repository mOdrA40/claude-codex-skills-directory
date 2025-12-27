/**
 * Standardized API Response Helpers
 */

export interface SuccessResponse<T> {
  success: true
  data: T
  meta?: {
    page?: number
    limit?: number
    total?: number
    totalPages?: number
  }
}

export interface ErrorResponse {
  success: false
  error: {
    code: string
    message: string
    details?: Record<string, string[]>
  }
}

export type ApiResponse<T> = SuccessResponse<T> | ErrorResponse

// Success responses
export const ok = <T>(data: T): SuccessResponse<T> => ({
  success: true,
  data,
})

export const created = <T>(data: T): SuccessResponse<T> => ({
  success: true,
  data,
})

export const paginated = <T>(
  data: T[],
  options: { page: number; limit: number; total: number }
): SuccessResponse<T[]> => ({
  success: true,
  data,
  meta: {
    page: options.page,
    limit: options.limit,
    total: options.total,
    totalPages: Math.ceil(options.total / options.limit),
  },
})

// Error responses
export const error = (code: string, message: string, details?: Record<string, string[]>): ErrorResponse => ({
  success: false,
  error: {
    code,
    message,
    ...(details && { details }),
  },
})
