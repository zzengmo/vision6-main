import { useLayoutEffect, useRef, useState } from 'react';
import { messageOf } from './api';

export function useInference<T>(key: string) {
  const [state, setState] = useState<{ key: string; result: T | null; running: boolean; error: string }>({ key, result: null, running: false, error: '' });
  if (state.key !== key) {
    setState({ key, result: null, running: false, error: '' });
  }
  const keyRef = useRef(key);
  const sequence = useRef(0);
  const request = useRef<AbortController | null>(null);
  useLayoutEffect(() => {
    keyRef.current = key;
    sequence.current += 1; request.current?.abort();
    return () => { sequence.current += 1; request.current?.abort(); };
  }, [key]);
  async function run(operation: (signal: AbortSignal) => Promise<T>) {
    request.current?.abort();
    const controller = new AbortController(); request.current = controller;
    const id = ++sequence.current, currentKey = keyRef.current;
    const current = () => sequence.current === id && keyRef.current === currentKey;
    setState({ key: currentKey, result: null, running: true, error: '' });
    const timer = setTimeout(() => controller.abort(), 240000);
    try {
      const value = await operation(controller.signal);
      if (current()) setState({ key: currentKey, result: value, running: false, error: '' });
    } catch (err) {
      if (current()) setState({ key: currentKey, result: null, running: false, error: controller.signal.aborted ? '요청 시간이 초과됐습니다. 서버 상태를 확인하고 다시 실행해 주세요.' : messageOf(err) });
    } finally { clearTimeout(timer); }
  }
  function cancel() {
    sequence.current += 1; request.current?.abort();
    setState({ key: keyRef.current, result: null, running: false, error: '' });
  }
  const visible = state.key === key ? state : { result: null, running: false, error: '' };
  return { result: visible.result, running: visible.running, error: visible.error, run, cancel };
}
