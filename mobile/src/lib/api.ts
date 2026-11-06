import Constants from 'expo-constants';
import { Platform } from 'react-native';

export function normalizeServer(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) throw new Error('PC 서버 주소를 입력해 주세요.');
  const url = new URL(/^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('http://PC주소:8504 형식으로 입력해 주세요.');
  }
  return url.origin;
}

export function defaultServer(): string {
  if (process.env.EXPO_PUBLIC_API_URL) return process.env.EXPO_PUBLIC_API_URL;
  const hostUri = Constants.expoConfig?.hostUri;
  const host = Platform.OS === 'web' && typeof window !== 'undefined'
    ? window.location.hostname : hostUri ? new URL(`http://${hostUri}`).hostname : '127.0.0.1';
  return `http://${host}:${process.env.EXPO_PUBLIC_API_PORT || '8504'}`;
}

export async function jsonResponse<T>(response: { text(): Promise<string>; ok: boolean; status: number }): Promise<T> {
  const text = await response.text();
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error('서버 응답을 읽을 수 없습니다. 서버 주소와 포트를 확인해 주세요.'); }
  if (!response.ok) throw new Error((value as { error?: string }).error || `요청 실패 (${response.status})`);
  return value as T;
}

export async function getJSON<T>(url: string, signal?: AbortSignal): Promise<T> {
  return jsonResponse<T>(await fetch(url, { signal }));
}

export function messageOf(error: unknown): string {
  if (error instanceof Error && !/network|fetch failed|failed to fetch|load failed/i.test(error.message)) return error.message;
  return 'PC 서버에 연결할 수 없습니다. 같은 Wi-Fi인지, 서버 실행 창이 열려 있는지 확인해 주세요.';
}
