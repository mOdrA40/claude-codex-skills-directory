/**
 * User Repository
 * 
 * Data access layer for user table.
 */

import { db } from "../db"
import { users, type NewUser, type User } from "../db/schema"
import { eq, count } from "drizzle-orm"

interface FindManyOptions {
  limit: number
  offset: number
}

export const userRepository = {
  async findMany(options: FindManyOptions): Promise<User[]> {
    return db.query.users.findMany({
      limit: options.limit,
      offset: options.offset,
      orderBy: (users, { desc }) => [desc(users.createdAt)],
    })
  },

  async findById(id: string): Promise<User | undefined> {
    return db.query.users.findFirst({
      where: eq(users.id, id),
    })
  },

  async findByEmail(email: string): Promise<User | undefined> {
    return db.query.users.findFirst({
      where: eq(users.email, email),
    })
  },

  async count(): Promise<number> {
    const result = await db.select({ count: count() }).from(users)
    return result[0]?.count ?? 0
  },

  async create(data: NewUser): Promise<User> {
    const [user] = await db.insert(users).values(data).returning()
    return user
  },

  async update(id: string, data: Partial<NewUser>): Promise<User | undefined> {
    const [user] = await db
      .update(users)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(users.id, id))
      .returning()
    return user
  },

  async delete(id: string): Promise<void> {
    await db.delete(users).where(eq(users.id, id))
  },
}
