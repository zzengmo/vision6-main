import { useState } from 'react';
import { Alert } from 'react-native';
import { shareResult } from '../lib/files';
import { messageOf } from '../lib/api';
import { Button } from './ui';

export function ShareButton({ uri, name, label }: { uri: string; name: string; label: string }) {
  const [busy, setBusy] = useState(false);
  async function share() {
    setBusy(true);
    try { await shareResult(uri, name); }
    catch (error) { Alert.alert('파일 공유', messageOf(error)); }
    finally { setBusy(false); }
  }
  return <Button small secondary busy={busy} onPress={() => void share()}>{label}</Button>;
}
