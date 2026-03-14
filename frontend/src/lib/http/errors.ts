export class ApiError extends Error {
  public status: number;
  public code: string;
  public requestId?: string;
  public detail?: unknown;

  constructor(
    message: string,
    status: number,
    code: string,
    requestId?: string,
    detail?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.detail = detail;
  }
}

export type ApiErrorResponse = {
  detail?: string | object;
  code?: string;
  message?: string;
};
