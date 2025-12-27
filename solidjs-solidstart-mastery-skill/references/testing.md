# Comprehensive Testing Strategies

## Testing Setup

### Vitest Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import solid from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solid()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,ts,tsx}'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
      ],
      thresholds: {
        branches: 80,
        functions: 80,
        lines: 80,
        statements: 80,
      },
    },
    deps: {
      optimizer: {
        web: {
          include: ['solid-js'],
        },
      },
    },
  },
  resolve: {
    conditions: ['development', 'browser'],
  },
});
```

### Test Setup File

```typescript
// src/test/setup.ts
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@solidjs/testing-library';
import { afterEach, vi } from 'vitest';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});
```

## Unit Testing

### Testing Signals

```typescript
// lib/signals/__tests__/createCounter.test.ts
import { describe, it, expect } from 'vitest';
import { createRoot } from 'solid-js';
import { createCounter } from '../createCounter';

describe('createCounter', () => {
  it('should initialize with default value', () => {
    createRoot(dispose => {
      const [count] = createCounter();
      expect(count()).toBe(0);
      dispose();
    });
  });

  it('should initialize with custom value', () => {
    createRoot(dispose => {
      const [count] = createCounter(10);
      expect(count()).toBe(10);
      dispose();
    });
  });

  it('should increment', () => {
    createRoot(dispose => {
      const [count, { increment }] = createCounter(0);
      increment();
      expect(count()).toBe(1);
      dispose();
    });
  });

  it('should decrement', () => {
    createRoot(dispose => {
      const [count, { decrement }] = createCounter(10);
      decrement();
      expect(count()).toBe(9);
      dispose();
    });
  });

  it('should reset to initial value', () => {
    createRoot(dispose => {
      const [count, { increment, reset }] = createCounter(5);
      increment();
      increment();
      expect(count()).toBe(7);
      reset();
      expect(count()).toBe(5);
      dispose();
    });
  });
});
```

### Testing Stores

```typescript
// lib/stores/__tests__/todoStore.test.ts
import { describe, it, expect } from 'vitest';
import { createRoot } from 'solid-js';
import { createTodoStore } from '../todoStore';

describe('createTodoStore', () => {
  it('should add todo', () => {
    createRoot(dispose => {
      const [state, actions] = createTodoStore();
      
      actions.addTodo('Test todo');
      
      expect(state.todos).toHaveLength(1);
      expect(state.todos[0].text).toBe('Test todo');
      expect(state.todos[0].completed).toBe(false);
      
      dispose();
    });
  });

  it('should toggle todo', () => {
    createRoot(dispose => {
      const [state, actions] = createTodoStore();
      
      actions.addTodo('Test todo');
      const id = state.todos[0].id;
      
      actions.toggleTodo(id);
      expect(state.todos[0].completed).toBe(true);
      
      actions.toggleTodo(id);
      expect(state.todos[0].completed).toBe(false);
      
      dispose();
    });
  });

  it('should delete todo', () => {
    createRoot(dispose => {
      const [state, actions] = createTodoStore();
      
      actions.addTodo('Todo 1');
      actions.addTodo('Todo 2');
      const id = state.todos[0].id;
      
      actions.deleteTodo(id);
      
      expect(state.todos).toHaveLength(1);
      expect(state.todos[0].text).toBe('Todo 2');
      
      dispose();
    });
  });

  it('should filter todos', () => {
    createRoot(dispose => {
      const [state, actions] = createTodoStore();
      
      actions.addTodo('Todo 1');
      actions.addTodo('Todo 2');
      actions.toggleTodo(state.todos[0].id);
      
      actions.setFilter('completed');
      expect(state.filteredTodos).toHaveLength(1);
      
      actions.setFilter('active');
      expect(state.filteredTodos).toHaveLength(1);
      
      actions.setFilter('all');
      expect(state.filteredTodos).toHaveLength(2);
      
      dispose();
    });
  });
});
```

## Component Testing

### Testing with Solid Testing Library

```typescript
// components/__tests__/Button.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@solidjs/testing-library';
import { Button } from '../ui/Button';

describe('Button', () => {
  it('renders children correctly', () => {
    render(() => <Button>Click me</Button>);
    expect(screen.getByRole('button')).toHaveTextContent('Click me');
  });

  it('calls onClick when clicked', async () => {
    const handleClick = vi.fn();
    render(() => <Button onClick={handleClick}>Click me</Button>);
    
    await fireEvent.click(screen.getByRole('button'));
    
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('is disabled when disabled prop is true', () => {
    render(() => <Button disabled>Click me</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('shows loading state', () => {
    render(() => <Button loading>Submit</Button>);
    
    expect(screen.getByRole('button')).toBeDisabled();
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('applies variant styles', () => {
    const { rerender } = render(() => <Button variant="primary">Button</Button>);
    expect(screen.getByRole('button')).toHaveClass('btn-primary');
    
    rerender(() => <Button variant="secondary">Button</Button>);
    expect(screen.getByRole('button')).toHaveClass('btn-secondary');
  });
});
```

### Testing Forms

```typescript
// components/__tests__/LoginForm.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@solidjs/testing-library';
import { LoginForm } from '../features/auth/LoginForm';

describe('LoginForm', () => {
  const mockOnSubmit = vi.fn();

  beforeEach(() => {
    mockOnSubmit.mockClear();
  });

  it('renders email and password fields', () => {
    render(() => <LoginForm onSubmit={mockOnSubmit} />);
    
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('shows validation errors for empty fields', async () => {
    render(() => <LoginForm onSubmit={mockOnSubmit} />);
    
    await fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    await waitFor(() => {
      expect(screen.getByText(/email is required/i)).toBeInTheDocument();
      expect(screen.getByText(/password is required/i)).toBeInTheDocument();
    });
    
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  it('shows error for invalid email', async () => {
    render(() => <LoginForm onSubmit={mockOnSubmit} />);
    
    await fireEvent.input(screen.getByLabelText(/email/i), {
      target: { value: 'invalid-email' },
    });
    await fireEvent.input(screen.getByLabelText(/password/i), {
      target: { value: 'password123' },
    });
    await fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    await waitFor(() => {
      expect(screen.getByText(/invalid email/i)).toBeInTheDocument();
    });
  });

  it('submits form with valid data', async () => {
    render(() => <LoginForm onSubmit={mockOnSubmit} />);
    
    await fireEvent.input(screen.getByLabelText(/email/i), {
      target: { value: 'user@example.com' },
    });
    await fireEvent.input(screen.getByLabelText(/password/i), {
      target: { value: 'password123' },
    });
    await fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith({
        email: 'user@example.com',
        password: 'password123',
      });
    });
  });

  it('disables button while submitting', async () => {
    mockOnSubmit.mockImplementation(() => new Promise(r => setTimeout(r, 100)));
    
    render(() => <LoginForm onSubmit={mockOnSubmit} />);
    
    await fireEvent.input(screen.getByLabelText(/email/i), {
      target: { value: 'user@example.com' },
    });
    await fireEvent.input(screen.getByLabelText(/password/i), {
      target: { value: 'password123' },
    });
    await fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    expect(screen.getByRole('button')).toBeDisabled();
    
    await waitFor(() => {
      expect(screen.getByRole('button')).not.toBeDisabled();
    });
  });
});
```

### Testing Async Components

```typescript
// components/__tests__/UserList.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@solidjs/testing-library';
import { QueryClient, QueryClientProvider } from '@tanstack/solid-query';
import { UserList } from '../features/users/UserList';
import * as api from '~/lib/api';

vi.mock('~/lib/api');

describe('UserList', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    vi.clearAllMocks();
  });

  const renderWithQuery = (component: () => JSX.Element) => {
    return render(() => (
      <QueryClientProvider client={queryClient}>
        {component()}
      </QueryClientProvider>
    ));
  };

  it('shows loading state', () => {
    vi.mocked(api.getUsers).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    renderWithQuery(() => <UserList />);
    
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('shows users when loaded', async () => {
    vi.mocked(api.getUsers).mockResolvedValue([
      { id: '1', name: 'John Doe', email: 'john@example.com' },
      { id: '2', name: 'Jane Doe', email: 'jane@example.com' },
    ]);

    renderWithQuery(() => <UserList />);

    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('Jane Doe')).toBeInTheDocument();
    });
  });

  it('shows error state', async () => {
    vi.mocked(api.getUsers).mockRejectedValue(new Error('Network error'));

    renderWithQuery(() => <UserList />);

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });

  it('shows empty state', async () => {
    vi.mocked(api.getUsers).mockResolvedValue([]);

    renderWithQuery(() => <UserList />);

    await waitFor(() => {
      expect(screen.getByText(/no users found/i)).toBeInTheDocument();
    });
  });
});
```

## Integration Testing

### Testing API Routes

```typescript
// routes/api/__tests__/users.test.ts
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { GET, POST, DELETE } from '../users';
import { db } from '~/lib/db';

vi.mock('~/lib/db');

describe('Users API', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('GET /api/users', () => {
    it('returns all users', async () => {
      const mockUsers = [
        { id: '1', name: 'John', email: 'john@example.com' },
      ];
      vi.mocked(db.user.findMany).mockResolvedValue(mockUsers);

      const request = new Request('http://localhost/api/users');
      const response = await GET({ request } as any);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data).toEqual(mockUsers);
    });

    it('supports pagination', async () => {
      vi.mocked(db.user.findMany).mockResolvedValue([]);

      const request = new Request('http://localhost/api/users?page=2&limit=10');
      await GET({ request } as any);

      expect(db.user.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          skip: 10,
          take: 10,
        })
      );
    });
  });

  describe('POST /api/users', () => {
    it('creates user with valid data', async () => {
      const newUser = { name: 'John', email: 'john@example.com', password: 'Password123' };
      const createdUser = { id: '1', ...newUser };
      
      vi.mocked(db.user.findUnique).mockResolvedValue(null);
      vi.mocked(db.user.create).mockResolvedValue(createdUser);

      const request = new Request('http://localhost/api/users', {
        method: 'POST',
        body: JSON.stringify(newUser),
      });
      const response = await POST({ request } as any);
      const data = await response.json();

      expect(response.status).toBe(201);
      expect(data.id).toBe('1');
    });

    it('returns 400 for invalid data', async () => {
      const request = new Request('http://localhost/api/users', {
        method: 'POST',
        body: JSON.stringify({ name: 'J' }), // Invalid: missing fields
      });
      const response = await POST({ request } as any);

      expect(response.status).toBe(400);
    });

    it('returns 409 for duplicate email', async () => {
      vi.mocked(db.user.findUnique).mockResolvedValue({ id: '1' } as any);

      const request = new Request('http://localhost/api/users', {
        method: 'POST',
        body: JSON.stringify({
          name: 'John',
          email: 'existing@example.com',
          password: 'Password123',
        }),
      });
      const response = await POST({ request } as any);

      expect(response.status).toBe(409);
    });
  });

  describe('DELETE /api/users/:id', () => {
    it('deletes user', async () => {
      vi.mocked(db.user.delete).mockResolvedValue({} as any);

      const request = new Request('http://localhost/api/users/1', {
        method: 'DELETE',
      });
      const response = await DELETE({ 
        request, 
        params: { id: '1' } 
      } as any);

      expect(response.status).toBe(204);
      expect(db.user.delete).toHaveBeenCalledWith({
        where: { id: '1' },
      });
    });
  });
});
```

## E2E Testing with Playwright

### Playwright Configuration

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
```

### E2E Test Examples

```typescript
// e2e/auth.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('should login successfully', async ({ page }) => {
    await page.goto('/login');
    
    await page.fill('[name="email"]', 'user@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.click('button[type="submit"]');
    
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('text=Welcome')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');
    
    await page.fill('[name="email"]', 'wrong@example.com');
    await page.fill('[name="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');
    
    await expect(page.locator('text=Invalid credentials')).toBeVisible();
  });

  test('should redirect to login for protected routes', async ({ page }) => {
    await page.goto('/dashboard');
    
    await expect(page).toHaveURL(/\/login/);
  });

  test('should logout successfully', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('[name="email"]', 'user@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.click('button[type="submit"]');
    
    // Logout
    await page.click('button:has-text("Logout")');
    
    await expect(page).toHaveURL('/');
    await page.goto('/dashboard');
    await expect(page).toHaveURL(/\/login/);
  });
});

// e2e/crud.spec.ts
test.describe('Todo CRUD', () => {
  test.beforeEach(async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('[name="email"]', 'user@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.click('button[type="submit"]');
    await page.goto('/todos');
  });

  test('should create todo', async ({ page }) => {
    await page.fill('[name="todo"]', 'New todo item');
    await page.click('button:has-text("Add")');
    
    await expect(page.locator('text=New todo item')).toBeVisible();
  });

  test('should toggle todo completion', async ({ page }) => {
    // Create todo
    await page.fill('[name="todo"]', 'Toggle me');
    await page.click('button:has-text("Add")');
    
    // Toggle
    await page.click('[data-testid="todo-checkbox"]');
    
    await expect(page.locator('.todo-item.completed')).toBeVisible();
  });

  test('should delete todo', async ({ page }) => {
    // Create todo
    await page.fill('[name="todo"]', 'Delete me');
    await page.click('button:has-text("Add")');
    
    // Delete
    await page.click('[data-testid="delete-todo"]');
    
    await expect(page.locator('text=Delete me')).not.toBeVisible();
  });
});
```

## Test Utils & Helpers

```typescript
// src/test/utils.tsx
import { render, RenderOptions } from '@solidjs/testing-library';
import { QueryClient, QueryClientProvider } from '@tanstack/solid-query';
import { Router } from '@solidjs/router';
import { ParentComponent, JSX } from 'solid-js';

interface ProvidersProps {
  queryClient?: QueryClient;
}

export function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });
}

export function createWrapper(options: ProvidersProps = {}): ParentComponent {
  const queryClient = options.queryClient ?? createTestQueryClient();

  return (props) => (
    <QueryClientProvider client={queryClient}>
      <Router>
        {props.children}
      </Router>
    </QueryClientProvider>
  );
}

export function renderWithProviders(
  ui: () => JSX.Element,
  options?: Omit<RenderOptions, 'wrapper'> & ProvidersProps
) {
  const { queryClient, ...renderOptions } = options ?? {};
  
  return render(ui, {
    wrapper: createWrapper({ queryClient }),
    ...renderOptions,
  });
}

// Mock factories
export function createMockUser(overrides = {}) {
  return {
    id: '1',
    name: 'Test User',
    email: 'test@example.com',
    createdAt: new Date().toISOString(),
    ...overrides,
  };
}

export function createMockTodo(overrides = {}) {
  return {
    id: crypto.randomUUID(),
    text: 'Test todo',
    completed: false,
    createdAt: new Date().toISOString(),
    ...overrides,
  };
}
```

## Test Coverage Script

```json
// package.json
{
  "scripts": {
    "test": "vitest",
    "test:watch": "vitest --watch",
    "test:coverage": "vitest run --coverage",
    "test:ui": "vitest --ui",
    "test:e2e": "playwright test",
    "test:e2e:ui": "playwright test --ui",
    "test:all": "npm run test:coverage && npm run test:e2e"
  }
}
```
