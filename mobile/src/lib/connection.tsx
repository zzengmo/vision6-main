import AsyncStorage from '@react-native-async-storage/async-storage';
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { defaultServer, getJSON, messageOf, normalizeServer } from './api';
import type { Metadata } from './types';

const STORAGE_KEY = 'steelvision.server.v1';
type Connection = { baseURL: string; metadata: Metadata | null; busy: boolean; error: string; identity: string; connect: (value: string) => Promise<void> };
const Context = createContext<Connection | null>(null);

export function ConnectionProvider({ children }: { children: ReactNode }) {
  const [baseURL, setBaseURL] = useState(defaultServer);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const request = useRef<AbortController | null>(null);
  const connect = useCallback(async (value: string) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true); setMetadata(null); setError(''); setRevision(r => r + 1);
    const timer = setTimeout(() => controller.abort(), 25000);
    try {
      const url = normalizeServer(value);
      setBaseURL(url);
      const health = await getJSON<{ ok: boolean; app: string }>(url + '/api/health', controller.signal);
      if (!health.ok || health.app !== 'steelvision-react') throw new Error('SteelVision 서버 주소인지 확인해 주세요.');
      const data = await getJSON<Metadata>(url + '/api/metadata', controller.signal);
      if (!Array.isArray(data.models) || !data.revision) throw new Error('서버의 모델 목록을 읽을 수 없습니다.');
      if (request.current !== controller) return;
      setMetadata(data);
      await AsyncStorage.setItem(STORAGE_KEY, url).catch(() => {});
    } catch (err) {
      if (request.current === controller) setError(controller.signal.aborted ? '연결 시간이 초과됐습니다. PC 주소와 Wi-Fi를 확인해 주세요.' : messageOf(err));
    } finally {
      clearTimeout(timer);
      if (request.current === controller) setBusy(false);
    }
  }, []);
  useEffect(() => {
    let active = true;
    void AsyncStorage.getItem(STORAGE_KEY).catch(() => null).then(saved => { if (active) void connect(saved || defaultServer()); });
    return () => { active = false; request.current?.abort(); };
  }, [connect]);
  return <Context.Provider value={{ baseURL, metadata, busy, error, identity: `${baseURL}|${metadata?.revision || ''}|${revision}`, connect }}>{children}</Context.Provider>;
}

export function useConnection(): Connection {
  const value = useContext(Context);
  if (!value) throw new Error('ConnectionProvider is missing');
  return value;
}
