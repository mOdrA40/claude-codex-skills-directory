# Advanced Design Patterns & Anti-Patterns

## Component Composition Patterns

### Compound Components

```typescript
// components/ui/Accordion.tsx
import { createContext, useContext, ParentComponent, Accessor } from 'solid-js';

interface AccordionContextValue {
  activeItem: Accessor<string | null>;
  setActiveItem: (id: string | null) => void;
  multiple: boolean;
  activeItems: Accessor<Set<string>>;
  toggleItem: (id: string) => void;
}

const AccordionContext = createContext<AccordionContextValue>();

export function useAccordion() {
  const ctx = useContext(AccordionContext);
  if (!ctx) throw new Error('useAccordion must be used within Accordion.Root');
  return ctx;
}

interface AccordionRootProps {
  multiple?: boolean;
  defaultValue?: string | string[];
}

const AccordionRoot: ParentComponent<AccordionRootProps> = (props) => {
  const [activeItem, setActiveItem] = createSignal<string | null>(
    typeof props.defaultValue === 'string' ? props.defaultValue : null
  );
  const [activeItems, setActiveItems] = createSignal<Set<string>>(
    new Set(Array.isArray(props.defaultValue) ? props.defaultValue : [])
  );

  const toggleItem = (id: string) => {
    if (props.multiple) {
      setActiveItems((prev) => {
        const next = new Set(prev);
        next.has(id) ? next.delete(id) : next.add(id);
        return next;
      });
    } else {
      setActiveItem((prev) => (prev === id ? null : id));
    }
  };

  return (
    <AccordionContext.Provider
      value={{
        activeItem,
        setActiveItem,
        multiple: props.multiple ?? false,
        activeItems,
        toggleItem,
      }}
    >
      <div class="accordion" role="region">{props.children}</div>
    </AccordionContext.Provider>
  );
};

const AccordionItem: ParentComponent<{ value: string }> = (props) => {
  const { activeItem, activeItems, multiple } = useAccordion();
  const isOpen = () =>
    multiple ? activeItems().has(props.value) : activeItem() === props.value;

  return (
    <div class="accordion-item" data-state={isOpen() ? 'open' : 'closed'}>
      {props.children}
    </div>
  );
};

const AccordionTrigger: ParentComponent<{ value: string }> = (props) => {
  const { toggleItem, activeItem, activeItems, multiple } = useAccordion();
  const isOpen = () =>
    multiple ? activeItems().has(props.value) : activeItem() === props.value;

  return (
    <button
      class="accordion-trigger"
      aria-expanded={isOpen()}
      onClick={() => toggleItem(props.value)}
    >
      {props.children}
      <span class="accordion-icon" data-state={isOpen() ? 'open' : 'closed'}>
        ▼
      </span>
    </button>
  );
};

const AccordionContent: ParentComponent<{ value: string }> = (props) => {
  const { activeItem, activeItems, multiple } = useAccordion();
  const isOpen = () =>
    multiple ? activeItems().has(props.value) : activeItem() === props.value;

  return (
    <Show when={isOpen()}>
      <div class="accordion-content" role="region">
        {props.children}
      </div>
    </Show>
  );
};

export const Accordion = {
  Root: AccordionRoot,
  Item: AccordionItem,
  Trigger: AccordionTrigger,
  Content: AccordionContent,
};

// Usage
<Accordion.Root multiple defaultValue={['item-1']}>
  <Accordion.Item value="item-1">
    <Accordion.Trigger value="item-1">Section 1</Accordion.Trigger>
    <Accordion.Content value="item-1">Content 1</Accordion.Content>
  </Accordion.Item>
  <Accordion.Item value="item-2">
    <Accordion.Trigger value="item-2">Section 2</Accordion.Trigger>
    <Accordion.Content value="item-2">Content 2</Accordion.Content>
  </Accordion.Item>
</Accordion.Root>
```

### Render Props / Function as Children

```typescript
// components/ClickOutside.tsx
interface ClickOutsideProps {
  onClickOutside: () => void;
  children: (ref: (el: HTMLElement) => void) => JSX.Element;
}

export function ClickOutside(props: ClickOutsideProps) {
  let containerRef: HTMLElement | undefined;

  const handleClick = (e: MouseEvent) => {
    if (containerRef && !containerRef.contains(e.target as Node)) {
      props.onClickOutside();
    }
  };

  onMount(() => document.addEventListener('mousedown', handleClick));
  onCleanup(() => document.removeEventListener('mousedown', handleClick));

  return props.children((el) => (containerRef = el));
}

// Usage
<ClickOutside onClickOutside={() => setIsOpen(false)}>
  {(ref) => (
    <div ref={ref} class="dropdown">
      <DropdownContent isOpen={isOpen()} />
    </div>
  )}
</ClickOutside>
```

### Polymorphic Components

```typescript
// components/ui/Box.tsx
import { JSX, splitProps, ValidComponent } from 'solid-js';
import { Dynamic } from 'solid-js/web';

type BoxProps<T extends ValidComponent> = {
  as?: T;
  class?: string;
  children?: JSX.Element;
} & JSX.IntrinsicElements[T extends keyof JSX.IntrinsicElements ? T : 'div'];

export function Box<T extends ValidComponent = 'div'>(props: BoxProps<T>) {
  const [local, others] = splitProps(props, ['as', 'class', 'children']);

  return (
    <Dynamic
      component={local.as || 'div'}
      class={local.class}
      {...others}
    >
      {local.children}
    </Dynamic>
  );
}

// Usage
<Box as="article" class="card">Content</Box>
<Box as="section" class="hero">Hero</Box>
<Box as="a" href="/about" class="link">About</Box>
```

## State Management Patterns

### Context + Store Pattern

```typescript
// lib/stores/app-store.tsx
import { createContext, useContext, ParentComponent } from 'solid-js';
import { createStore, produce } from 'solid-js/store';

interface AppState {
  theme: 'light' | 'dark';
  sidebar: { isOpen: boolean; width: number };
  notifications: Notification[];
}

interface AppActions {
  toggleTheme: () => void;
  toggleSidebar: () => void;
  setSidebarWidth: (width: number) => void;
  addNotification: (n: Omit<Notification, 'id'>) => void;
  removeNotification: (id: string) => void;
}

type AppStore = [AppState, AppActions];

const AppContext = createContext<AppStore>();

export function useAppStore() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppStore must be used within AppProvider');
  return ctx;
}

export const AppProvider: ParentComponent = (props) => {
  const [state, setState] = createStore<AppState>({
    theme: 'light',
    sidebar: { isOpen: true, width: 280 },
    notifications: [],
  });

  const actions: AppActions = {
    toggleTheme: () =>
      setState('theme', (t) => (t === 'light' ? 'dark' : 'light')),

    toggleSidebar: () =>
      setState('sidebar', 'isOpen', (open) => !open),

    setSidebarWidth: (width) =>
      setState('sidebar', 'width', width),

    addNotification: (n) =>
      setState(produce((s) => {
        s.notifications.push({ ...n, id: crypto.randomUUID() });
      })),

    removeNotification: (id) =>
      setState('notifications', (list) => list.filter((n) => n.id !== id)),
  };

  return (
    <AppContext.Provider value={[state, actions]}>
      {props.children}
    </AppContext.Provider>
  );
};

// Usage
function Header() {
  const [state, actions] = useAppStore();

  return (
    <header>
      <button onClick={actions.toggleSidebar}>
        {state.sidebar.isOpen ? 'Close' : 'Open'} Sidebar
      </button>
      <button onClick={actions.toggleTheme}>
        Theme: {state.theme}
      </button>
    </header>
  );
}
```

### Signal Factory Pattern

```typescript
// lib/signals/createLocalStorage.ts
import { createSignal, createEffect, Accessor, Setter } from 'solid-js';
import { isServer } from 'solid-js/web';

interface CreateLocalStorageOptions<T> {
  key: string;
  defaultValue: T;
  serializer?: {
    read: (raw: string) => T;
    write: (value: T) => string;
  };
}

export function createLocalStorage<T>(
  options: CreateLocalStorageOptions<T>
): [Accessor<T>, Setter<T>] {
  const {
    key,
    defaultValue,
    serializer = {
      read: JSON.parse,
      write: JSON.stringify,
    },
  } = options;

  const getInitialValue = (): T => {
    if (isServer) return defaultValue;
    try {
      const item = localStorage.getItem(key);
      return item ? serializer.read(item) : defaultValue;
    } catch {
      return defaultValue;
    }
  };

  const [value, setValue] = createSignal<T>(getInitialValue());

  createEffect(() => {
    if (isServer) return;
    try {
      localStorage.setItem(key, serializer.write(value()));
    } catch (e) {
      console.error('Failed to save to localStorage:', e);
    }
  });

  return [value, setValue];
}

// Usage
const [theme, setTheme] = createLocalStorage({
  key: 'app-theme',
  defaultValue: 'light' as const,
});

const [user, setUser] = createLocalStorage<User | null>({
  key: 'current-user',
  defaultValue: null,
});
```

### Async Signal Pattern

```typescript
// lib/signals/createAsyncSignal.ts
import { createSignal, Accessor } from 'solid-js';

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

interface AsyncActions<T> {
  execute: () => Promise<void>;
  reset: () => void;
  setData: (data: T) => void;
}

type UseAsync<T> = [Accessor<AsyncState<T>>, AsyncActions<T>];

export function createAsyncSignal<T>(
  asyncFn: () => Promise<T>,
  options?: { immediate?: boolean }
): UseAsync<T> {
  const [state, setState] = createSignal<AsyncState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = async () => {
    setState({ data: null, loading: true, error: null });
    try {
      const data = await asyncFn();
      setState({ data, loading: false, error: null });
    } catch (error) {
      setState({ data: null, loading: false, error: error as Error });
    }
  };

  const reset = () => {
    setState({ data: null, loading: false, error: null });
  };

  const setData = (data: T) => {
    setState((prev) => ({ ...prev, data }));
  };

  if (options?.immediate) {
    execute();
  }

  return [state, { execute, reset, setData }];
}

// Usage
function UserProfile() {
  const [userState, { execute: loadUser }] = createAsyncSignal(
    () => api.getUser(userId),
    { immediate: true }
  );

  return (
    <Switch>
      <Match when={userState().loading}>
        <Spinner />
      </Match>
      <Match when={userState().error}>
        <Error error={userState().error!} onRetry={loadUser} />
      </Match>
      <Match when={userState().data}>
        <UserCard user={userState().data!} />
      </Match>
    </Switch>
  );
}
```

## Anti-Patterns to Avoid

### ❌ Derived State in Effects

```typescript
// ❌ BAD: Anti-pattern
const [count, setCount] = createSignal(0);
const [double, setDouble] = createSignal(0);

createEffect(() => {
  setDouble(count() * 2); // WRONG! Creates unnecessary re-renders
});

// ✅ GOOD: Use createMemo
const [count, setCount] = createSignal(0);
const double = createMemo(() => count() * 2);
```

### ❌ Accessing Signals During Setup

```typescript
// ❌ BAD: Signal accessed outside reactive context
function BadComponent(props: { userId: string }) {
  const userId = props.userId; // Not reactive!
  const [user] = createResource(() => fetchUser(userId));
  // ...
}

// ✅ GOOD: Access in reactive context
function GoodComponent(props: { userId: string }) {
  const [user] = createResource(
    () => props.userId, // Reactive accessor
    fetchUser
  );
  // ...
}
```

### ❌ Storing Derived State

```typescript
// ❌ BAD: Unnecessary signal for derived state
function BadComponent() {
  const [items, setItems] = createSignal<Item[]>([]);
  const [filteredItems, setFilteredItems] = createSignal<Item[]>([]);
  const [filter, setFilter] = createSignal('');

  createEffect(() => {
    setFilteredItems(
      items().filter((i) => i.name.includes(filter()))
    );
  });
}

// ✅ GOOD: Use createMemo
function GoodComponent() {
  const [items, setItems] = createSignal<Item[]>([]);
  const [filter, setFilter] = createSignal('');

  const filteredItems = createMemo(() =>
    items().filter((i) => i.name.includes(filter()))
  );
}
```

### ❌ Props Spreading Without splitProps

```typescript
// ❌ BAD: Can override critical props
function BadButton(props: ButtonProps) {
  return <button class="btn" {...props} />; // class can be overwritten!
}

// ✅ GOOD: Use splitProps and mergeProps
import { splitProps, mergeProps } from 'solid-js';

function GoodButton(props: ButtonProps) {
  const [local, others] = splitProps(props, ['class', 'variant']);
  const merged = mergeProps({ variant: 'primary' }, props);

  return (
    <button
      class={`btn btn-${merged.variant} ${local.class ?? ''}`}
      {...others}
    />
  );
}
```

### ❌ Missing Cleanup in Effects

```typescript
// ❌ BAD: Memory leak
createEffect(() => {
  const interval = setInterval(() => {
    console.log('tick');
  }, 1000);
  // Missing cleanup!
});

// ✅ GOOD: Proper cleanup
createEffect(() => {
  const interval = setInterval(() => {
    console.log('tick');
  }, 1000);

  onCleanup(() => clearInterval(interval));
});
```

## Advanced Patterns

### Deferred Value (Performance)

```typescript
// hooks/createDeferredValue.ts
import { createSignal, createEffect, Accessor } from 'solid-js';

export function createDeferredValue<T>(
  value: Accessor<T>,
  delay = 200
): Accessor<T> {
  const [deferred, setDeferred] = createSignal<T>(value());

  createEffect(() => {
    const current = value();
    const timeout = setTimeout(() => setDeferred(() => current), delay);
    onCleanup(() => clearTimeout(timeout));
  });

  return deferred;
}

// Usage - prevents expensive re-renders during typing
function Search() {
  const [query, setQuery] = createSignal('');
  const deferredQuery = createDeferredValue(query, 300);

  const results = createMemo(() =>
    expensiveSearch(deferredQuery()) // Only runs after 300ms pause
  );
}
```

### Batch Updates Pattern

```typescript
import { batch } from 'solid-js';

function handleFormSubmit() {
  // ✅ All updates happen in single batch
  batch(() => {
    setLoading(true);
    setErrors([]);
    setFormData(initialData);
  });

  // Instead of 3 separate re-renders, only 1
}
```

### Untrack Pattern

```typescript
import { untrack } from 'solid-js';

createEffect(() => {
  // This will track `count` changes
  const currentCount = count();

  // This won't cause effect to re-run when `name` changes
  const currentName = untrack(() => name());

  console.log(`${currentName}: ${currentCount}`);
});
```

## Module Organization

### Feature-Slice Design

```
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   │   ├── LoginForm.tsx
│   │   │   └── RegisterForm.tsx
│   │   ├── hooks/
│   │   │   └── useAuth.ts
│   │   ├── stores/
│   │   │   └── auth-store.ts
│   │   ├── api/
│   │   │   └── auth-api.ts
│   │   └── index.ts  # Public exports only
│   ├── users/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── index.ts
│   └── posts/
│       └── ...
├── shared/
│   ├── ui/          # Shared UI components
│   ├── lib/         # Shared utilities
│   └── types/       # Shared types
└── routes/          # Page components only
```

```typescript
// features/auth/index.ts - Clean public API
export { LoginForm } from './components/LoginForm';
export { RegisterForm } from './components/RegisterForm';
export { useAuth } from './hooks/useAuth';
export { AuthProvider } from './stores/auth-store';
export type { User, AuthState } from './types';
```
