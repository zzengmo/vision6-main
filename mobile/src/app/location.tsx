import { useState } from 'react';
import { Text, View } from 'react-native';
import { useConnection } from '../lib/connection';
import { useInference } from '../lib/useInference';
import { upload, validatePhoto } from '../lib/files';
import type { Localization, Photo } from '../lib/types';
import { PhotoInput } from '../components/PhotoInput';
import { ShareButton } from '../components/ShareButton';
import { Button, Card, Notice, Page, Preview, styles } from '../components/ui';

export default function Location() {
  const { baseURL, metadata, identity } = useConnection();
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [view, setView] = useState<'original_url' | 'overlay_url' | 'mask_url'>('overlay_url');
  const invalidSize = photo && (photo.width !== 1600 || photo.height !== 256);
  const { result, running, error, run, cancel } = useInference<Localization>(`${identity}|${photo?.uri}`);
  function localize() {
    if (!photo) return;
    void run(signal => { validatePhoto(photo, true); return upload<Localization>(baseURL + '/api/localize', photo, signal); });
  }
  return <Page eyebrow="DEFECT LOCALIZATION" title="결함이 있는 위치까지." subtitle="U-Net이 예측한 마스크와 원본 좌표를 확인하세요.">
    <Notice>1600×256 원본 JPG·PNG를 사용하세요. 품질 판정과 별도 모델이며, 결함·배경의 이진 마스크를 제공합니다.</Notice>
    <PhotoInput photo={photo} onChange={setPhoto} disabled={running}/>
    <Card title="2. U-Net 위치 분석">
      <Text style={styles.body}>저장 마스크 임계값 {metadata?.segmentation?.threshold.toFixed(4) || '—'}</Text>
      <Text style={styles.caption}>원래 검증 기준을 유지합니다. 표시 박스는 최대 2개입니다.</Text>
      {!!invalidSize && <Notice error>선택한 사진은 {photo.width}×{photo.height}입니다. 위치 분석에는 1600×256 원본이나 예시 사진을 사용해 주세요.</Notice>}
      <Button disabled={!photo || !metadata?.segmentation || !!invalidSize} busy={running} onPress={localize}>{running ? '실제 U-Net으로 분석 중…' : '결함 위치 찾기'}</Button>
      {running && <Button secondary onPress={cancel}>요청 취소</Button>}
      {!!error && <Notice error>{error}</Notice>}
    </Card>
    {result && <>
      <Card title="실제 예측 결과">
        <View style={styles.row}>{([['original_url', '원본'], ['overlay_url', '오버레이'], ['mask_url', '이진 마스크']] as const).map(([key, label]) => <Button small key={key} secondary={view !== key} onPress={() => setView(key)}>{label}</Button>)}</View>
        <Preview uri={result[view]} label={view === 'original_url' ? '원본' : view === 'mask_url' ? '이진 예측 마스크' : 'U-Net 예측 마스크와 위치 박스'} />
        <View style={styles.row}><View style={styles.stat}><Text style={styles.caption}>표시 박스</Text><Text style={styles.statValue}>{result.boxes.length}개</Text></View><View style={styles.stat}><Text style={styles.caption}>마스크 픽셀</Text><Text style={styles.statValue}>{result.mask_pixels.toLocaleString()}</Text></View><View style={styles.stat}><Text style={styles.caption}>전체 처리</Text><Text style={styles.statValue}>{result.elapsed_seconds.toFixed(2)}초</Text></View></View>
        <Notice>정상 사진에도 영역이 표시될 수 있습니다. 이 결과는 결함 유형 분류나 품질 검수 모델의 판정과 구분해 주세요.</Notice>
      </Card>
      <Card title="원본 기준 좌표">
        {!result.boxes.length && <Text style={styles.body}>표시할 박스가 없습니다.</Text>}
        {result.boxes.map((box, index) => <View key={index} style={styles.stat}><Text style={styles.cardTitle}>박스 {index + 1}</Text><Text style={styles.body}>x {box.x} · y {box.y}{'\n'}너비 {box.width} · 높이 {box.height}</Text><Text style={styles.caption}>영역 평균 점수 {box.mean_score.toFixed(4)}</Text></View>)}
        <Text style={styles.caption}>좌표는 선택한 원본 1600×256 이미지 기준입니다.</Text>
      </Card>
      <Card title="결과 저장·공유"><View style={styles.row}>
        <ShareButton uri={result.overlay_url} name="unet_overlay.png" label="결과 PNG"/><ShareButton uri={result.mask_url} name="unet_mask.png" label="마스크 PNG"/><ShareButton uri={result.csv_url} name="unet_boxes.csv" label="좌표 CSV"/>
      </View></Card>
    </>}
  </Page>;
}
