import { useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import { useConnection } from '../lib/connection';
import { useInference } from '../lib/useInference';
import { upload, validatePhoto } from '../lib/files';
import type { Classification, Photo } from '../lib/types';
import { PhotoInput } from '../components/PhotoInput';
import { ShareButton } from '../components/ShareButton';
import { Button, Card, Notice, Page, Preview, colors, styles } from '../components/ui';

export default function Inspection() {
  const { baseURL, metadata, identity } = useConnection();
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [modelId, setModelId] = useState('resnet18_tta');
  const [policyId, setPolicyId] = useState('RF_same_FP_budget');
  const model = metadata?.models.find(m => m.id === modelId) || metadata?.models[0];
  const policy = model?.policies.find(p => p.id === policyId) || model?.policies[0];
  const { result, running, error, run, cancel } = useInference<Classification>(`${identity}|${photo?.uri}|${model?.id}|${policy?.id}`);
  function inspect() {
    if (!photo || !model || !policy) return;
    void run(signal => {
      validatePhoto(photo, false);
      return upload<Classification>(`${baseURL}/api/inspect?model=${encodeURIComponent(model.id)}&policy=${encodeURIComponent(policy.id)}`, photo, signal);
    });
  }
  return <Page eyebrow="QUALITY INSPECTION" title="작은 차이를, 정확하게." subtitle="사진 한 장으로 불량 의심 여부를 확인하세요.">
    <PhotoInput photo={photo} onChange={setPhoto} disabled={running} />
    <Card title="2. 검수 모델과 판정 기준">
      {!metadata && <Notice>하단 연결 설정에서 PC 서버를 연결해 주세요.</Notice>}
      {metadata?.models.map(m => <Pressable accessibilityRole="radio" accessibilityState={{ checked: model?.id === m.id, disabled: running || !m.ready }} key={m.id} disabled={running || !m.ready} onPress={() => { setModelId(m.id); setPolicyId(m.policies[0]?.id || 'checkpoint'); }} style={{ borderWidth: 1, borderColor: model?.id === m.id ? colors.blue : colors.line, borderRadius: 12, padding: 13, gap: 5, backgroundColor: model?.id === m.id ? colors.pale : 'white', opacity: m.ready ? 1 : .45 }}>
        <Text style={[styles.body, { fontWeight: '600' }]}>{model?.id === m.id ? '● ' : '○ '}{m.label}</Text><Text style={styles.caption}>{m.ready ? m.input : '모델 준비 필요'}</Text>
      </Pressable>)}
      {(model?.policies.length || 0) > 1 && <Text style={styles.cardTitle}>운영점</Text>}
      {model?.policies.map(p => <Pressable key={p.id} accessibilityRole="radio" accessibilityState={{ checked: policy?.id === p.id }} disabled={running} onPress={() => setPolicyId(p.id)} style={{ paddingVertical: 9, gap: 3 }}><Text style={styles.body}>{policy?.id === p.id ? '● ' : '○ '}{p.label}</Text><Text style={styles.caption}>임계값 {p.threshold.toPrecision(7)}</Text></Pressable>)}
      <Button disabled={!photo || !model?.ready || !policy} busy={running} onPress={inspect}>{running ? '실제 모델로 분석 중…' : '품질 검수 실행'}</Button>
      {running && <Button secondary onPress={cancel}>요청 취소</Button>}
      {!!error && <Notice error>{error}</Notice>}
    </Card>
    {result ? <Card title="검사 결과">
      <Text accessibilityLiveRegion="polite" style={{ fontSize: 32, fontWeight: '700', color: result.decision === 'defect' ? colors.orange : colors.green }}>{result.decision === 'defect' ? '불량 의심' : '정상'}</Text>
      <View style={styles.row}><View style={styles.stat}><Text style={styles.caption}>불량 점수</Text><Text style={styles.statValue}>{result.score.toFixed(6)}</Text></View><View style={styles.stat}><Text style={styles.caption}>모델 처리 시간</Text><Text style={styles.statValue}>{result.elapsed_seconds.toFixed(3)}초</Text></View></View>
      <Text style={styles.caption}>판정 임계값 {result.threshold.toPrecision(8)} · {result.model_id}</Text>
      <Preview uri={result.original_url} label="검사 원본" />
      <Notice>이미지 전체의 이진 판정입니다. 점수는 보정된 실제 불량 확률이 아니며, 결함 위치는 별도 탭에서 분석합니다.</Notice>
      <View style={styles.row}><ShareButton uri={result.csv_url} name="inspection.csv" label="판정 CSV 공유"/><ShareButton uri={result.original_url} name="inspection_original.png" label="원본 PNG 공유"/></View>
    </Card> : !running && <Notice>사진을 선택하고 검수를 실행하면 결과가 여기에 표시됩니다.</Notice>}
  </Page>;
}
