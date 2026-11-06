import { useLayoutEffect, useRef, useState } from 'react';
import { Text, View } from 'react-native';
import { useConnection } from '../lib/connection';
import { messageOf } from '../lib/api';
import { pickDocument, pickGallery, samplePhoto } from '../lib/files';
import type { Photo } from '../lib/types';
import { Button, Card, Notice, Preview, styles } from './ui';

export function PhotoInput({ photo, onChange, disabled = false }: { photo: Photo | null; onChange: (photo: Photo | null) => void; disabled?: boolean }) {
  const { baseURL, metadata, identity } = useConnection();
  const identityRef = useRef(identity);
  useLayoutEffect(() => { identityRef.current = identity; }, [identity]);
  const [busy, setBusy] = useState(false), [error, setError] = useState('');
  async function choose(action: () => Promise<Photo | null>) {
    const current = identityRef.current;
    setBusy(true); setError('');
    try { const value = await action(); if (value && identityRef.current === current) onChange(value); }
    catch (err) { setError(messageOf(err)); }
    finally { setBusy(false); }
  }
  const blocked = disabled || busy;
  return <Card title="1. 검사 사진">
    <View style={styles.row}><Button disabled={blocked} secondary onPress={() => void choose(pickGallery)}>사진 보관함</Button><Button disabled={blocked} secondary onPress={() => void choose(pickDocument)}>파일 선택</Button></View>
    <Text style={styles.caption}>JPG·PNG, 최대 30MB · HEIC 사진은 JPG로 변환합니다.</Text>
    <View style={styles.row}>
      <Button small disabled={blocked || !metadata} secondary onPress={() => void choose(() => samplePhoto(baseURL, '/api/samples/normal', 'normal.png'))}>정상 예시</Button>
      <Button small disabled={blocked || !metadata} secondary onPress={() => void choose(() => samplePhoto(baseURL, '/api/samples/defect', 'defect.png'))}>불량 예시</Button>
      {metadata?.examples.some(e => e.filename === '1fc012f23.jpg' && e.available) && <Button small disabled={blocked} secondary onPress={() => void choose(() => samplePhoto(baseURL, '/api/examples/1fc012f23.jpg', '1fc012f23.jpg'))}>제공 사진</Button>}
    </View>
    {busy && <Notice>사진을 불러오고 있습니다.</Notice>}
    {!!error && <Notice error>{error}</Notice>}
    {photo && <><Text style={styles.body}>{photo.name}</Text><Text style={styles.caption}>{photo.width}×{photo.height}{photo.size ? ` · ${(photo.size / 1024 / 1024).toFixed(2)}MB` : ''}</Text><Preview uri={photo.uri} label="선택한 원본"/><Button small secondary disabled={blocked} onPress={() => onChange(null)}>선택 해제</Button></>}
  </Card>;
}
