import { ApiError, ApiErrorResponse } from './errors';

// Read backend URL from env, fallback to relative path (proxy) or localhost
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined | null>;
  timeout?: number;
}

/**
 * Shared fetch wrapper
 */
export async function apiClient<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { params, timeout = 15000, ...customConfig } = options;

  let urlText = endpoint.startsWith('http')
    ? endpoint
    : `${API_BASE_URL}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;

  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    });
    const qs = searchParams.toString();
    if (qs) {
      urlText += `?${qs}`;
    }
  }

  // Handle Token
  // In a real app, you might want to read a cookie or localStorage here
  let token = '';
  if (typeof window !== 'undefined') {
    token = localStorage.getItem('agframe_token') || '';
  }

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...customConfig.headers,
  };

  const config: RequestInit = {
    ...customConfig,
    headers,
  };

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  config.signal = controller.signal;

  try {
    const response = await fetch(urlText, config);
    clearTimeout(id);

    // Handle 401 Unauthorized globally if needed (e.g. dispatch event or redirect)
    if (response.status === 401 && typeof window !== 'undefined') {
      localStorage.removeItem('agframe_token');
      // window.location.href = '/login'; // Optional: auto redirect on 401
    }

    if (!response.ok) {
      const requestId = response.headers.get('x-request-id') || undefined;
      let errorData: ApiErrorResponse = {};
      
      try {
        errorData = await response.json();
      } catch {
        // failed to parse JSON error
      }

      throw new ApiError(
        errorData.message || 'An error occurred during the request.',
        response.status,
        errorData.code || 'UNKNOWN_ERROR',
        requestId,
        errorData.detail
      );
    }

    if (response.status === 204) {
      return {} as T;
    }

    return await response.json();
  } catch (error) {
    clearTimeout(id);
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('Request timed out', 408, 'TIMEOUT');
    }
    throw new ApiError(
      error instanceof Error ? error.message : 'Unknown network error',
      0,
      'NETWORK_ERROR'
    );
  }
}
