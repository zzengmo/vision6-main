import { useState } from 'react';
import { Text, TextInput, View } from 'react-native';
import { useConnection } from '../lib/connection';
import { defaultServer } from '../lib/api';
import { Button, Card, Notice, Page, colors, styles } from '../components/ui';

export default function Settings() {
  const { baseURL, metadata, busy, connect } = useConnection();
  const [editedAddress, setDraft] = useState<string | null>(null);
  const draft = editedAddress ?? baseURL;
  return <Page eyebrow="CONNECTION" title="PC와 연결하세요." subtitle="사진은 PC 서버에서 분석하고, 결과는 휴대폰에서 확인합니다.">
    <Card title="서버 주소">
      <Text style={styles.caption}>PC 실행 창에 표시된 모바일 API 주소</Text>
      <TextInput accessibilityLabel="서버 주소" placeholder="http://192.168.0.10:8504" autoCapitalize="none" autoCorrect={false} keyboardType="url" value={draft} onChangeText={setDraft} editable={!busy} style={styles.input} onSubmitEditing={() => void connect(draft)} />
      <Button busy={busy} onPress={() => void connect(draft)}>{busy ? '모델 목록 확인 중…' : '연결 확인'}</Button>
      <Button small secondary disabled={busy} onPress={() => { const suggested = defaultServer(); setDraft(suggested); void connect(suggested); }}>자동 주소로 다시 연결</Button>
      {metadata && <Notice>연결됐습니다. 사용 가능한 분류 모델 {metadata.models.filter(m => m.ready).length}개 · U-Net {metadata.segmentation ? '연결됨' : '준비 필요'}</Notice>}
    </Card>
    <Card title="아이폰에서 시작하기">
      {['PC에서 Start_SteelVision_Expo.cmd를 실행합니다.', 'PC와 아이폰을 같은 Wi-Fi에 연결합니다.', '아이폰 카메라로 실행 창의 QR을 읽어 Expo Go에서 엽니다.', '로컬 네트워크 접근 요청이 나오면 허용합니다.'].map((text, i) => <View key={text} style={{ flexDirection: 'row', gap: 12 }}><Text style={{ color: colors.blue, fontWeight: '700' }}>{i + 1}.</Text><Text style={[styles.body, { flex: 1 }]}>{text}</Text></View>)}
      <Text style={styles.caption}>휴대폰에서 127.0.0.1은 휴대폰 자신을 가리킵니다. 연결이 안 되면 PC의 Wi-Fi IP 주소를 입력해 주세요.</Text>
    </Card>
    <Card title="입력과 결과 안내">
      <Text style={styles.body}>품질 검수: JPG·PNG, 30MB·2,500만 픽셀 이하{ '\n'}위치 분석: 1600×256 원본{ '\n'}사진 보관함의 HEIC: 크기를 유지해 JPG로 변환</Text>
      <Text style={styles.caption}>모델은 PC에서 실행하므로 검사 중 PC 서버가 켜져 있어야 합니다. PNG·CSV 공유 버튼에서 ‘파일에 저장’을 선택할 수 있습니다.</Text>
    </Card>
  </Page>;
}
