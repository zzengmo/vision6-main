import { useState, type ReactNode } from 'react';
import { ActivityIndicator, Image, Modal, Pressable, ScrollView, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { useConnection } from '../lib/connection';

export const colors = { navy: '#0A344A', blue: '#087CA7', bg: '#F3F7FA', ink: '#193C50', muted: '#647E8F', line: '#DDE8EE', pale: '#EAF4F8', orange: '#AE6814', green: '#147B62' };
export const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { padding: 20, gap: 18, paddingBottom: 40, width: '100%', maxWidth: 760, alignSelf: 'center' },
  eyebrow: { color: colors.blue, fontSize: 11, fontWeight: '700', letterSpacing: 2 },
  heading: { color: colors.ink, fontSize: 29, fontWeight: '700', lineHeight: 39 },
  body: { color: colors.ink, fontSize: 14, lineHeight: 23 },
  caption: { color: colors.muted, fontSize: 12, lineHeight: 20 },
  row: { flexDirection: 'row', gap: 10, alignItems: 'center', flexWrap: 'wrap' },
  card: { backgroundColor: 'white', padding: 18, borderRadius: 20, borderWidth: 1, borderColor: colors.line, gap: 14 },
  cardTitle: { color: colors.ink, fontSize: 16, fontWeight: '700' },
  input: { borderWidth: 1, borderColor: colors.line, backgroundColor: 'white', borderRadius: 12, padding: 14, color: colors.ink, fontSize: 14 },
  stat: { flex: 1, minWidth: 90, padding: 13, borderRadius: 12, backgroundColor: colors.bg, gap: 5 },
  statValue: { color: colors.ink, fontSize: 20, fontWeight: '700' },
});

export function Button({ children, onPress, disabled = false, secondary = false, busy = false, small = false }: { children: ReactNode; onPress: () => void; disabled?: boolean; secondary?: boolean; busy?: boolean; small?: boolean }) {
  return <Pressable accessibilityRole="button" disabled={disabled || busy} onPress={onPress} style={({ pressed }) => ({ minHeight: small ? 40 : 50, paddingHorizontal: small ? 13 : 18, paddingVertical: 12, borderRadius: 12, borderWidth: 1, borderColor: secondary ? colors.line : colors.blue, backgroundColor: secondary ? 'white' : colors.blue, opacity: disabled || busy ? .45 : pressed ? .7 : 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 9 })}>
    {busy && <ActivityIndicator size="small" color={secondary ? colors.blue : 'white'} />}
    <Text style={{ color: secondary ? colors.ink : 'white', fontWeight: '600', fontSize: small ? 12 : 14 }}>{children}</Text>
  </Pressable>;
}

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return <View style={styles.card}>{title && <Text style={styles.cardTitle}>{title}</Text>}{children}</View>;
}

export function Notice({ children, error = false }: { children: ReactNode; error?: boolean }) {
  return <View accessibilityLiveRegion="polite" style={{ backgroundColor: error ? '#FFF3E3' : colors.pale, padding: 14, borderRadius: 12 }}><Text style={{ color: error ? colors.orange : colors.ink, fontSize: 12, lineHeight: 21 }}>{children}</Text></View>;
}

export function Page({ eyebrow, title, subtitle, children }: { eyebrow: string; title: string; subtitle: string; children: ReactNode }) {
  const { metadata, busy, error } = useConnection();
  return <SafeAreaView edges={['top']} style={styles.screen}>
    <View style={{ backgroundColor: colors.navy, paddingHorizontal: 22, paddingVertical: 18, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
      <Text style={{ color: 'white', fontSize: 23, fontWeight: '700', letterSpacing: -1 }}>steelvision <Text style={{ fontSize: 10, letterSpacing: 1 }}>ML</Text></Text>
      <Pressable accessibilityRole="button" accessibilityLabel="서버 연결 설정" onPress={() => router.push('/settings')}><Text style={{ color: metadata ? '#9BE2CA' : '#F4CF9E', fontSize: 12 }}>{busy ? '연결 중' : metadata ? '● 서버 연결됨' : '○ 연결 필요'}</Text></Pressable>
    </View>
    <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={styles.content}>
      <View style={{ gap: 7 }}><Text style={styles.eyebrow}>{eyebrow}</Text><Text style={styles.heading}>{title}</Text><Text style={styles.caption}>{subtitle}</Text></View>
      {!!error && <Notice error>{error}{'\n'}하단 ‘연결 설정’에서 PC 주소를 확인해 주세요.</Notice>}
      {children}
      <Text style={[styles.caption, { textAlign: 'center', paddingTop: 12 }]}>STEELVISION ML · 모바일 품질 검수</Text>
    </ScrollView>
  </SafeAreaView>;
}

export function Preview({ uri, label }: { uri: string; label: string }) {
  const [open, setOpen] = useState(false);
  const { width } = useWindowDimensions();
  return <View style={{ gap: 7 }}>
    <Text style={styles.caption}>{label}</Text>
    <Pressable accessibilityRole="button" accessibilityLabel={`${label} 확대`} onPress={() => setOpen(true)} style={{ borderRadius: 12, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.bg, overflow: 'hidden' }}>
      <Image source={{ uri }} resizeMode="contain" style={{ width: '100%', height: 140 }} />
      <Text style={[styles.caption, { textAlign: 'right', padding: 8, fontSize: 10 }]}>눌러서 확대</Text>
    </Pressable>
    <Modal visible={open} animationType="fade" onRequestClose={() => setOpen(false)}>
      <SafeAreaView style={{ flex: 1, backgroundColor: '#081C28', padding: 16 }}>
        <Button secondary onPress={() => setOpen(false)}>닫기</Button>
        <ScrollView maximumZoomScale={6} minimumZoomScale={1} centerContent contentContainerStyle={{ flexGrow: 1, justifyContent: 'center' }}>
          <Image source={{ uri }} resizeMode="contain" style={{ width: width - 32, height: Math.max(220, (width - 32) / 6.25) }} />
        </ScrollView>
        <Text style={{ color: 'white', fontSize: 12, textAlign: 'center', padding: 12 }}>아이폰에서는 두 손가락으로 확대할 수 있습니다.</Text>
      </SafeAreaView>
    </Modal>
  </View>;
}
