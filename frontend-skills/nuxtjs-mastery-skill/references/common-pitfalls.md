# Common Pitfalls & Bugs — Vue.js & Nuxt 3

> 20 tahun pengalaman = 20 tahun membuat bug dan belajar dari kesalahan. Ini adalah kumpulan bug yang PASTI akan kamu temui.

## Table of Contents
1. [Reactivity Pitfalls](#reactivity-pitfalls)
2. [Lifecycle Pitfalls](#lifecycle-pitfalls)
3. [SSR/Hydration Pitfalls](#ssrhydration-pitfalls)
4. [TanStack Query Pitfalls](#tanstack-query-pitfalls)
5. [TypeScript Pitfalls](#typescript-pitfalls)
6. [Memory Leaks](#memory-leaks)
7. [Race Conditions](#race-conditions)
8. [Performance Killers](#performance-killers)

---

## Reactivity Pitfalls

### 1. Destructuring Reactive Objects

```typescript
// ❌ BUG: Reactivity hilang!
const store = useMyStore()
const { count, name } = store // count dan name TIDAK reactive!

// Kenapa? Destructuring membuat copy primitive values

// ✅ SOLUSI 1: Gunakan storeToRefs
import { storeToRefs } from 'pinia'
const { count, name } = storeToRefs(store) // Tetap reactive!

// ✅ SOLUSI 2: Akses langsung
const count = computed(() => store.count)

// ✅ SOLUSI 3: toRef untuk single property
const count = toRef(store, 'count')
```

### 2. Mutating Props

```typescript
// ❌ BUG: Mutating props akan error!
const props = defineProps<{ items: string[] }>()

function addItem(item: string) {
  props.items.push(item) // 💥 Vue warns: Avoid mutating a prop directly
}

// ✅ SOLUSI: Emit event ke parent
const emit = defineEmits<{
  'update:items': [items: string[]]
}>()

function addItem(item: string) {
  emit('update:items', [...props.items, item])
}

// Parent component
<ChildComponent v-model:items="items" />
```

### 3. Array/Object Mutation

```typescript
// ❌ BUG: Direct mutation tidak trigger reactivity di beberapa kasus
const list = ref([1, 2, 3])

// Ini TIDAK akan trigger update di Vue 2 (Vue 3 sudah fix)
list.value[0] = 999

// ❌ BUG yang masih terjadi di Vue 3:
const obj = reactive({ a: 1 })
obj.b = 2 // Works
delete obj.a // Works

// Tapi object dengan Map/Set behavior beda
const map = reactive(new Map())
map.set('key', 'value') // ✅ Works
map.delete('key') // ✅ Works

// ✅ BEST PRACTICE: Selalu replace entire array/object untuk clarity
list.value = [...list.value.slice(0, 0), 999, ...list.value.slice(1)]
// Atau lebih simple:
list.value = list.value.map((item, i) => i === 0 ? 999 : item)
```

### 4. Computed Side Effects

```typescript
// ❌ BUG: Side effects di computed
const items = ref([3, 1, 2])
const sorted = computed(() => {
  console.log('Computing...') // Side effect!
  items.value.sort() // Mutating original! 💥
  return items.value
})

// Computed akan infinite loop atau inconsistent state

// ✅ SOLUSI: Pure function, no side effects
const sorted = computed(() => {
  return [...items.value].sort() // Clone first!
})

// Untuk side effects, gunakan watch
watch(items, (newItems) => {
  console.log('Items changed:', newItems)
})
```

### 5. toRaw/markRaw Misuse

```typescript
// ❌ BUG: Menggunakan toRaw terlalu sering
const data = ref({ count: 0 })
const raw = toRaw(data.value)
raw.count = 1 // TIDAK trigger reactivity!

// ❌ BUG: markRaw di data yang seharusnya reactive
const user = markRaw({ name: 'John', role: 'admin' })
const state = reactive({ user })
state.user.name = 'Jane' // TIDAK trigger update!

// ✅ Gunakan toRaw HANYA untuk:
// 1. Passing ke external library yang tidak support Proxy
// 2. Deep clone untuk comparison
// 3. Serialization (JSON.stringify)

// ✅ Gunakan markRaw HANYA untuk:
// 1. Large objects yang tidak perlu reactive (chart data, etc.)
// 2. External library instances (axios, lodash, etc.)
const chartInstance = markRaw(new Chart())
```

---

## Lifecycle Pitfalls

### 1. Async onMounted

```typescript
// ❌ BUG: Error handling hilang
onMounted(async () => {
  const data = await fetchData() // Jika error, tidak ada handler!
  state.value = data
})

// ✅ SOLUSI: Proper error handling
onMounted(async () => {
  try {
    const data = await fetchData()
    state.value = data
  } catch (error) {
    errorState.value = error as Error
    toast.error('Failed to load data')
  }
})

// ✅ LEBIH BAIK: Gunakan TanStack Query
const { data, error, isLoading } = useQuery({
  queryKey: ['data'],
  queryFn: fetchData,
})
```

### 2. Using DOM Before Mount

```typescript
// ❌ BUG: DOM belum ada
<script setup>
const el = document.querySelector('.my-element') // 💥 null!
</script>

// ✅ SOLUSI 1: onMounted
onMounted(() => {
  const el = document.querySelector('.my-element') // ✅
})

// ✅ SOLUSI 2: Template ref
const myElement = ref<HTMLElement | null>(null)
onMounted(() => {
  console.log(myElement.value) // ✅ Element tersedia
})

<template>
  <div ref="myElement" class="my-element">Content</div>
</template>
```

### 3. Watch Timing

```typescript
// ❌ BUG: Watch tidak trigger untuk initial value
const data = ref('initial')

watch(data, (newVal) => {
  console.log('Changed:', newVal) // TIDAK log 'initial'!
})

// ✅ SOLUSI: immediate option
watch(data, (newVal) => {
  console.log('Changed:', newVal)
}, { immediate: true })

// ✅ ATAU: watchEffect untuk auto-track
watchEffect(() => {
  console.log('Data:', data.value) // Auto-run di awal
})
```

---

## SSR/Hydration Pitfalls

### 1. Browser-Only APIs

```typescript
// ❌ BUG: window/document di SSR
const screenWidth = window.innerWidth // 💥 window is not defined!
localStorage.setItem('key', 'value') // 💥 localStorage is not defined!

// ✅ SOLUSI 1: Check environment
const screenWidth = ref(0)
if (import.meta.client) {
  screenWidth.value = window.innerWidth
}

// ✅ SOLUSI 2: onMounted (hanya di client)
onMounted(() => {
  screenWidth.value = window.innerWidth
})

// ✅ SOLUSI 3: VueUse (auto-handle SSR)
import { useWindowSize, useLocalStorage } from '@vueuse/core'
const { width } = useWindowSize()
const stored = useLocalStorage('key', 'default')
```

### 2. Different Render Output

```typescript
// ❌ BUG: Server dan client menghasilkan HTML berbeda
<template>
  <div>{{ Date.now() }}</div> <!-- Berbeda di server vs client! -->
  <div>{{ Math.random() }}</div> <!-- Berbeda! -->
</template>

// ✅ SOLUSI 1: ClientOnly wrapper
<template>
  <ClientOnly>
    <div>{{ Date.now() }}</div>
  </ClientOnly>
</template>

// ✅ SOLUSI 2: Set di onMounted
const timestamp = ref<number | null>(null)
onMounted(() => {
  timestamp.value = Date.now()
})
```

### 3. Conditional Rendering Mismatch

```typescript
// ❌ BUG: v-if berdasarkan client-only value
const isLoggedIn = ref(false)
onMounted(() => {
  isLoggedIn.value = !!localStorage.getItem('token')
})

<template>
  <!-- Server render: false, Client render: true = MISMATCH! -->
  <div v-if="isLoggedIn">Dashboard</div>
</template>

// ✅ SOLUSI: Gunakan useCookie atau handle state properly
const token = useCookie('auth-token')
const isLoggedIn = computed(() => !!token.value)

// ✅ ATAU: Tampilkan loading state
const isHydrated = ref(false)
onMounted(() => {
  isHydrated.value = true
})

<template>
  <div v-if="!isHydrated">Loading...</div>
  <div v-else-if="isLoggedIn">Dashboard</div>
</template>
```

---

## TanStack Query Pitfalls

### 1. Query Key Changes

```typescript
// ❌ BUG: Object baru di setiap render = infinite refetch!
const { data } = useQuery({
  queryKey: ['products', { category, page }], // Object baru setiap render!
  queryFn: fetchProducts,
})

// ✅ SOLUSI: Stabilize query key dengan computed
const queryKey = computed(() => ['products', { 
  category: category.value, 
  page: page.value 
}])

const { data } = useQuery({
  queryKey,
  queryFn: fetchProducts,
})
```

### 2. Stale Closure in queryFn

```typescript
// ❌ BUG: Stale closure - filters tidak update
const filters = ref({ category: 'all' })

const { data } = useQuery({
  queryKey: ['products'],
  queryFn: () => fetchProducts(filters.value), // 💥 Stale closure!
})

// filters.value berubah, tapi queryFn masih pakai nilai lama

// ✅ SOLUSI: Include dependency di queryKey
const { data } = useQuery({
  queryKey: computed(() => ['products', filters.value]),
  queryFn: () => fetchProducts(filters.value),
})
```

### 3. Missing Error Handling

```typescript
// ❌ BUG: Error tidak ditangani
const { data } = useQuery({
  queryKey: ['products'],
  queryFn: fetchProducts,
})

// Jika error, app bisa crash tanpa feedback ke user

// ✅ SOLUSI: Selalu handle error state
const { data, error, isError } = useQuery({
  queryKey: ['products'],
  queryFn: fetchProducts,
})

<template>
  <ErrorMessage v-if="isError" :error="error" />
  <ProductList v-else-if="data" :products="data" />
</template>
```

### 4. Forgetting to Invalidate

```typescript
// ❌ BUG: Data stale setelah mutation
const { mutate } = useMutation({
  mutationFn: createProduct,
  onSuccess: () => {
    toast.success('Created!') // Data di list masih lama!
  },
})

// ✅ SOLUSI: Invalidate related queries
const queryClient = useQueryClient()

const { mutate } = useMutation({
  mutationFn: createProduct,
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['products'] })
    toast.success('Created!')
  },
})
```

---

## TypeScript Pitfalls

### 1. Type Assertion Abuse

```typescript
// ❌ BUG: Type assertion menutupi bug
const data = await fetch('/api/users') as User[] // 💥 data bukan User[]!

// ✅ SOLUSI: Proper type guard atau validation
const response = await fetch('/api/users')
const data: unknown = await response.json()

// Validate dengan Zod
import { z } from 'zod'
const UserSchema = z.object({
  id: z.string(),
  name: z.string(),
})
const users = z.array(UserSchema).parse(data)
```

### 2. Non-null Assertion

```typescript
// ❌ BUG: Non-null assertion (!) berbahaya
const user = users.find(u => u.id === id)!
console.log(user.name) // 💥 Runtime error jika tidak ketemu!

// ✅ SOLUSI: Handle undefined case
const user = users.find(u => u.id === id)
if (!user) {
  throw new Error(`User ${id} not found`)
}
console.log(user.name) // TypeScript happy, runtime safe
```

### 3. Generic Type Loss

```typescript
// ❌ BUG: Generic type hilang
function process<T>(items: T[]): T[] {
  return items.map(item => {
    return { ...item } // Type jadi object, bukan T!
  })
}

// ✅ SOLUSI: Preserve type
function process<T extends object>(items: T[]): T[] {
  return items.map(item => ({ ...item }) as T)
}
```

---

## Memory Leaks

### 1. Event Listeners

```typescript
// ❌ LEAK: Listener tidak di-remove
onMounted(() => {
  window.addEventListener('resize', handleResize)
})
// Component unmount, listener masih ada!

// ✅ SOLUSI: Cleanup di onUnmounted
onMounted(() => {
  window.addEventListener('resize', handleResize)
})
onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})

// ✅ LEBIH BAIK: VueUse auto-cleanup
import { useEventListener } from '@vueuse/core'
useEventListener('resize', handleResize) // Auto cleanup!
```

### 2. Timers dan Intervals

```typescript
// ❌ LEAK: Interval tidak di-clear
onMounted(() => {
  setInterval(pollData, 5000) // Terus jalan setelah unmount!
})

// ✅ SOLUSI: Store dan clear
const intervalId = ref<NodeJS.Timeout>()

onMounted(() => {
  intervalId.value = setInterval(pollData, 5000)
})

onUnmounted(() => {
  if (intervalId.value) clearInterval(intervalId.value)
})

// ✅ LEBIH BAIK: VueUse
import { useIntervalFn } from '@vueuse/core'
const { pause, resume } = useIntervalFn(pollData, 5000)
```

### 3. Subscriptions

```typescript
// ❌ LEAK: Subscription tidak di-unsubscribe
const subscription = observable.subscribe(handleData)
// Subscription terus aktif!

// ✅ SOLUSI: Unsubscribe di cleanup
onUnmounted(() => {
  subscription.unsubscribe()
})
```

---

## Race Conditions

### 1. Stale Response

```typescript
// ❌ BUG: Response lama override response baru
let data = ref(null)

async function fetchData(id: string) {
  const response = await fetch(`/api/items/${id}`)
  data.value = await response.json() // Response lama bisa arrive belakangan!
}

// User klik item 1, lalu klik item 2
// Response item 1 arrive setelah item 2
// data.value = item 1 (SALAH!)

// ✅ SOLUSI: Abort previous request
const controller = ref<AbortController | null>(null)

async function fetchData(id: string) {
  controller.value?.abort() // Cancel previous
  controller.value = new AbortController()
  
  try {
    const response = await fetch(`/api/items/${id}`, {
      signal: controller.value.signal,
    })
    data.value = await response.json()
  } catch (e) {
    if (e.name !== 'AbortError') throw e
  }
}

// ✅ LEBIH BAIK: TanStack Query handle ini automatically!
const { data } = useQuery({
  queryKey: ['item', id],
  queryFn: () => fetchItem(id),
})
```

### 2. Concurrent Mutations

```typescript
// ❌ BUG: Double submit
const isSubmitting = ref(false)

async function handleSubmit() {
  const data = await submitForm() // User double-click = 2 submissions!
  router.push('/success')
}

// ✅ SOLUSI: Guard dengan loading state
async function handleSubmit() {
  if (isSubmitting.value) return
  isSubmitting.value = true
  
  try {
    await submitForm()
    router.push('/success')
  } finally {
    isSubmitting.value = false
  }
}

// ✅ LEBIH BAIK: Disable button di template
<button :disabled="isSubmitting" @click="handleSubmit">
  {{ isSubmitting ? 'Submitting...' : 'Submit' }}
</button>
```

---

## Performance Killers

### 1. Unnecessary Reactivity

```typescript
// ❌ SLOW: Reactive di data besar yang tidak berubah
const hugeData = reactive(arrayOf10000Items) // Proxy overhead!

// ✅ SOLUSI: markRaw atau shallowRef
import { markRaw, shallowRef } from 'vue'
const hugeData = shallowRef(arrayOf10000Items) // Shallow reactivity
const chartConfig = markRaw(complexChartConfig) // No reactivity
```

### 2. Computed Without Caching

```typescript
// ❌ SLOW: Method di template = recalculate setiap render
<template>
  <div v-for="item in items" :key="item.id">
    {{ formatPrice(item.price) }} <!-- Called N times per render! -->
  </div>
</template>

// ✅ SOLUSI: Precompute atau cached computed
const formattedItems = computed(() => 
  items.value.map(item => ({
    ...item,
    formattedPrice: formatPrice(item.price)
  }))
)
```

### 3. Deep Watch

```typescript
// ❌ SLOW: Deep watch di object besar
watch(hugeObject, callback, { deep: true }) // Check semua nested properties!

// ✅ SOLUSI: Watch specific paths
watch(
  () => hugeObject.value.users.length,
  callback
)

// Atau watch multiple specific values
watch(
  [() => obj.value.a, () => obj.value.b],
  ([newA, newB]) => { /* ... */ }
)
```

### 4. v-if vs v-show Misuse

```typescript
// ❌ SLOW: v-if untuk toggle yang sering
<Modal v-if="isOpen" /> <!-- Re-create DOM setiap toggle! -->

// ✅ Gunakan v-show untuk frequent toggles
<Modal v-show="isOpen" /> <!-- Hanya toggle display:none -->

// ✅ Gunakan v-if untuk:
// - Expensive components yang jarang shown
// - Conditional data fetching (v-if prevents mount)
// - Components dengan lifecycle side effects
```
