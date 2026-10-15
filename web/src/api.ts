export interface LogFile {
  name: string;
  kind: "metrics" | "spans";
  stream: string | null;
  rank: number;
}

export interface Run {
  id: string;
  name: string;
  project: string;
  bytes: number;
  modified: number;
  files: LogFile[];
}

export interface RunDetails {
  run: Run;
  config: Record<string, unknown>;
  summary: Record<string, unknown>;
  metrics: string[];
  rows: number;
  warnings: string[];
}

export interface History {
  x: number[];
  series: {metric: string; values: (number | null)[]; nonfinite: number}[];
  total: number;
  warnings: string[];
}

export interface Records {
  records: Record<string, unknown>[];
  next: string | null;
  warnings: string[];
}

export function apiUrl(path: string, params: Record<string, string | number | undefined> = {}): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) query.set(key, String(value));
  }
  return `${path}${query.size ? `?${query}` : ""}`;
}

export async function get<T>(path: string, params: Record<string, string | number | undefined> = {}, signal?: AbortSignal): Promise<T> {
  const response = await fetch(apiUrl(path, params), {signal, cache: "no-store"});
  if (!response.ok) {
    const body: {error?: string} = await response.json();
    throw new Error(body.error || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function display(value: unknown): string {
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (typeof record["$et.nonfinite"] === "string") return record["$et.nonfinite"];
    if (typeof record["$et.integer"] === "string") return record["$et.integer"];
    if (typeof record["$et.number"] === "string") return record["$et.number"];
    return JSON.stringify(value, null, 2);
  }
  return value === null ? "null" : String(value);
}

export function align(histories: History[], metric: string): {x: number[]; ys: (number | null)[][]} {
  const x = [...new Set(histories.flatMap(history => history.x))].sort((a, b) => a - b);
  const ys = histories.map(history => {
    const values = history.series.find(series => series.metric === metric)?.values;
    const points = new Map(history.x.map((axis, index) => [axis, values?.[index] ?? null]));
    return x.map(axis => points.get(axis) ?? null);
  });
  return {x, ys};
}
