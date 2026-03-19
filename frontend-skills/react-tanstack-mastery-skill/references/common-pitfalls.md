# Common Pitfalls & Bugs

## Table of Contents
1. [Stale Closure Bugs](#stale-closure-bugs)
2. [Race Conditions](#race-conditions)
3. [Infinite Re-renders](#infinite-re-renders)
4. [Query Invalidation Mistakes](#query-invalidation-mistakes)
5. [Hydration Mismatches](#hydration-mismatches)
6. [Memory Leaks](#memory-leaks)
7. [TypeScript Pitfalls](#typescript-pitfalls)

## Stale Closure Bugs

**Paling umum dan sulit di-debug!**

```typescript
// ❌ BUG: Stale closure - count selalu 0
function Counter() {
  const [count, setCount] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      console.log(count) // Selalu 0!
      setCount(count + 1) // Selalu set ke 1
    }, 1000)
    return () => clearInterval(interval)
  }, []) // Empty deps = closure captures initial count

  return <div>{count}</div>
}

// ✅ FIX 1: Functional update
useEffect(() => {
  const interval = setInterval(() => {
    setCount(prev => prev + 1) // Use previous value
  }, 1000)
  return () => clearInterval(interval)
}, [])

// ✅ FIX 2: Include in deps + useCallback
const increment = useCallback(() => {
  setCount(prev => prev + 1)
}, [])

useEffect(() => {
  const interval = setInterval(increment, 1000)
  return () => clearInterval(interval)
}, [increment])

// ✅ FIX 3: useRef untuk mutable value
function Counter() {
  const [count, setCount] = useState(0)
  const countRef = useRef(count)
  
  useEffect(() => {
    countRef.current = count
  }, [count])

  useEffect(() => {
    const interval = setInterval(() => {
      console.log(countRef.current) // Always current!
      setCount(prev => prev + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [])
}

// ❌ BUG: Event handler dengan stale state
function SearchInput() {
  const [query, setQuery] = useState('')

  const handleSearch = () => {
    // query bisa stale jika function dibuat sekali
    fetch(`/api/search?q=${query}`)
  }

  return (
    <>
      <input value={query} onChange={e => setQuery(e.target.value)} />
      <button onClick={handleSearch}>Search</button>
    </>
  )
}

// ✅ FIX: Inline handler atau useCallback dengan deps
const handleSearch = useCallback(() => {
  fetch(`/api/search?q=${query}`)
}, [query])
```

## Race Conditions

```typescript
// ❌ BUG: Race condition - response lama bisa override yang baru
function UserProfile({ userId }) {
  const [user, setUser] = useState(null)

  useEffect(() => {
    fetchUser(userId).then(setUser)
  }, [userId])
  // Jika userId berubah cepat: User A request → User B request → User A response arrives last → shows User A!
}

// ✅ FIX 1: AbortController
useEffect(() => {
  const controller = new AbortController()
  
  fetchUser(userId, { signal: controller.signal })
    .then(setUser)
    .catch(err => {
      if (err.name !== 'AbortError') throw err
    })
  
  return () => controller.abort()
}, [userId])

// ✅ FIX 2: Request ID tracking
useEffect(() => {
  let cancelled = false
  
  fetchUser(userId).then(data => {
    if (!cancelled) setUser(data)
  })
  
  return () => { cancelled = true }
}, [userId])

// ✅ BEST: TanStack Query (handles automatically)
function UserProfile({ userId }) {
  const { data: user } = useQuery({
    queryKey: ['user', userId],
    queryFn: () => fetchUser(userId),
  })
  // Race conditions handled internally!
}
```

## Infinite Re-renders

```typescript
// ❌ BUG: Object created on every render triggers useEffect
function Component() {
  const config = { theme: 'dark' } // New object every render!

  useEffect(() => {
    applyConfig(config)
  }, [config]) // Always different reference → infinite loop
}

// ✅ FIX 1: useMemo
const config = useMemo(() => ({ theme: 'dark' }), [])

// ✅ FIX 2: Move outside component
const config = { theme: 'dark' } // Stable reference

function Component() {
  useEffect(() => {
    applyConfig(config)
  }, [config])
}

// ❌ BUG: setState in useEffect without deps
useEffect(() => {
  setCount(count + 1) // Infinite loop!
})

// ❌ BUG: Derived state anti-pattern
function UserList({ users }) {
  const [filteredUsers, setFilteredUsers] = useState([])
  
  useEffect(() => {
    setFilteredUsers(users.filter(u => u.active)) // Unnecessary!
  }, [users])
}

// ✅ FIX: Derive during render
function UserList({ users }) {
  const filteredUsers = useMemo(
    () => users.filter(u => u.active),
    [users]
  )
}

// ❌ BUG: Callback that changes every render
function Parent() {
  const handleClick = () => console.log('clicked') // New function every render

  return <MemoizedChild onClick={handleClick} /> // Memoization broken!
}

// ✅ FIX: useCallback
const handleClick = useCallback(() => console.log('clicked'), [])
```

## Query Invalidation Mistakes

```typescript
// ❌ BUG: Wrong invalidation scope
const updateUser = useMutation({
  mutationFn: (data) => api.updateUser(data),
  onSuccess: () => {
    queryClient.invalidateQueries(['users']) // Invalidates ALL user queries
  },
})

// ✅ FIX: Precise invalidation
onSuccess: (_, { userId }) => {
  // Only invalidate specific user
  queryClient.invalidateQueries(['users', userId])
  // Or invalidate list only
  queryClient.invalidateQueries({ queryKey: ['users'], exact: true })
}

// ❌ BUG: Invalidation before mutation complete
const createUser = useMutation({
  mutationFn: (data) => api.createUser(data),
  onMutate: () => {
    queryClient.invalidateQueries(['users']) // Too early!
  },
})

// ✅ FIX: Invalidate in onSuccess
onSuccess: () => {
  queryClient.invalidateQueries(['users'])
}

// ❌ BUG: Missing query key hierarchy
// Keys: ['users', 'list'], ['users', 'detail', '123']
queryClient.invalidateQueries(['users', 'list']) // Doesn't invalidate details!

// ✅ FIX: Use hierarchical keys properly
// invalidateQueries(['users']) invalidates ALL users/* queries
```

## Hydration Mismatches

```typescript
// ❌ BUG: Different content server vs client
function DateDisplay() {
  return <span>{new Date().toLocaleString()}</span>
  // Server renders at server time, client hydrates at client time!
}

// ✅ FIX: Client-only rendering
function DateDisplay() {
  const [date, setDate] = useState<Date | null>(null)
  
  useEffect(() => {
    setDate(new Date())
  }, [])
  
  if (!date) return <span>Loading...</span>
  return <span>{date.toLocaleString()}</span>
}

// ❌ BUG: Using browser APIs during SSR
function WindowSize() {
  const [width, setWidth] = useState(window.innerWidth) // Error on server!
}

// ✅ FIX: Check for window
const [width, setWidth] = useState(
  typeof window !== 'undefined' ? window.innerWidth : 0
)

// ❌ BUG: Random IDs
function Input() {
  const id = `input-${Math.random()}` // Different on server vs client!
  return <input id={id} />
}

// ✅ FIX: useId hook
import { useId } from 'react'

function Input() {
  const id = useId() // Same on server and client
  return <input id={id} />
}

// ❌ BUG: Conditional rendering based on localStorage
function ThemeProvider({ children }) {
  const theme = localStorage.getItem('theme') // Error on server!
  return <ThemeContext.Provider value={theme}>{children}</ThemeContext.Provider>
}

// ✅ FIX: Initialize safely
const [theme, setTheme] = useState('light') // Default for SSR

useEffect(() => {
  setTheme(localStorage.getItem('theme') ?? 'light')
}, [])
```

## Memory Leaks

```typescript
// ❌ LEAK: Uncleared subscriptions
useEffect(() => {
  const unsubscribe = eventBus.on('event', handler)
  // Missing cleanup!
}, [])

// ✅ FIX
useEffect(() => {
  const unsubscribe = eventBus.on('event', handler)
  return () => unsubscribe()
}, [])

// ❌ LEAK: Growing arrays in state
function Chat() {
  const [messages, setMessages] = useState([])

  useEffect(() => {
    socket.on('message', (msg) => {
      setMessages(prev => [...prev, msg]) // Unbounded growth!
    })
  }, [])
}

// ✅ FIX: Limit array size
setMessages(prev => [...prev, msg].slice(-100)) // Keep last 100

// ❌ LEAK: Closure over large objects
useEffect(() => {
  const largeData = fetchLargeData()
  
  return () => {
    // largeData still referenced in closure even after cleanup!
  }
}, [])

// ✅ FIX: Clear references
useEffect(() => {
  let largeData = fetchLargeData()
  
  return () => {
    largeData = null // Allow garbage collection
  }
}, [])
```

## TypeScript Pitfalls

```typescript
// ❌ BUG: Type assertion without validation
const user = data as User // data might not be User!

// ✅ FIX: Runtime validation
import { z } from 'zod'
const userSchema = z.object({
  id: z.string(),
  name: z.string(),
})
const user = userSchema.parse(data)

// ❌ BUG: Optional chaining hides bugs
const name = user?.profile?.name ?? 'Unknown'
// If user.profile is always supposed to exist, this hides the bug!

// ✅ FIX: Be explicit about expectations
if (!user.profile) {
  throw new Error('User profile is missing')
}
const name = user.profile.name

// ❌ BUG: Non-null assertion
const element = document.getElementById('app')!
element.innerHTML = 'Hello' // Crash if element doesn't exist!

// ✅ FIX: Handle null case
const element = document.getElementById('app')
if (!element) {
  throw new Error('App element not found')
}
element.innerHTML = 'Hello'

// ❌ BUG: Using any
function processData(data: any) {
  return data.items.map(item => item.name) // Runtime error if wrong shape!
}

// ✅ FIX: Proper typing
interface DataResponse {
  items: Array<{ name: string }>
}
function processData(data: DataResponse) {
  return data.items.map(item => item.name)
}

// ❌ BUG: Event handler types
<input onChange={(e) => setName(e.target.value)} />
// e.target could be null in strict mode!

// ✅ FIX: Type guard
<input 
  onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value)
  }} 
/>
```

## TanStack-Specific Pitfalls

```typescript
// ❌ BUG: Creating queryFn inline
useQuery({
  queryKey: ['user', userId],
  queryFn: async () => {
    const response = await fetch(`/api/users/${userId}`)
    return response.json()
  },
})
// queryFn recreated every render, but Query handles this.
// Real issue: not extracting to reusable query options

// ✅ FIX: Query options factory
const userQueries = {
  detail: (id: string) => queryOptions({
    queryKey: ['user', id],
    queryFn: () => fetchUser(id),
  }),
}

// ❌ BUG: Mutation without error handling
const mutation = useMutation({
  mutationFn: createUser,
  onSuccess: () => {
    toast.success('Created!')
  },
  // onError missing - user sees nothing on failure!
})

// ✅ FIX: Always handle errors
const mutation = useMutation({
  mutationFn: createUser,
  onSuccess: () => toast.success('Created!'),
  onError: (error) => toast.error(error.message),
})

// ❌ BUG: staleTime di query bukan di queryClient
// Setiap query override default, inconsistent caching
useQuery({
  queryKey: ['users'],
  queryFn: fetchUsers,
  staleTime: 60000, // Repeated in every useQuery call
})

// ✅ FIX: Set defaults in queryClient atau query options factory
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes default
    },
  },
})
```
