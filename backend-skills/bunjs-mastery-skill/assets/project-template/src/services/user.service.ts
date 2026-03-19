/**
 * User Service
 * 
 * Business logic layer for user operations.
 */

import { userRepository } from "../repositories/user.repository"
import { NotFoundError, ConflictError } from "../utils/errors"

interface CreateUserInput {
  email: string
  name: string
  password: string
}

interface UpdateUserInput {
  email?: string
  name?: string
}

interface PaginationOptions {
  page: number
  limit: number
}

export const userService = {
  async list(options: PaginationOptions) {
    const offset = (options.page - 1) * options.limit
    const [users, total] = await Promise.all([
      userRepository.findMany({ limit: options.limit, offset }),
      userRepository.count(),
    ])

    return { users, total }
  },

  async getById(id: string) {
    const user = await userRepository.findById(id)

    if (!user) {
      throw new NotFoundError("User", id)
    }

    return user
  },

  async create(input: CreateUserInput) {
    // Check for existing email
    const existing = await userRepository.findByEmail(input.email)
    if (existing) {
      throw new ConflictError("Email already registered")
    }

    // Hash password
    const hashedPassword = await Bun.password.hash(input.password, {
      algorithm: "argon2id",
    })

    // Create user
    const user = await userRepository.create({
      ...input,
      password: hashedPassword,
    })

    // Don't return password
    const { password: _, ...userWithoutPassword } = user
    return userWithoutPassword
  },

  async update(id: string, input: UpdateUserInput) {
    // Check user exists
    const existing = await userRepository.findById(id)
    if (!existing) {
      throw new NotFoundError("User", id)
    }

    // Check email uniqueness if updating email
    if (input.email && input.email !== existing.email) {
      const emailExists = await userRepository.findByEmail(input.email)
      if (emailExists) {
        throw new ConflictError("Email already in use")
      }
    }

    const user = await userRepository.update(id, input)

    const { password: _, ...userWithoutPassword } = user!
    return userWithoutPassword
  },

  async delete(id: string) {
    const existing = await userRepository.findById(id)
    if (!existing) {
      throw new NotFoundError("User", id)
    }

    await userRepository.delete(id)
  },
}
