import { Tabs } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Text } from 'react-native';
import { ConnectionProvider } from '../lib/connection';
import { colors } from '../components/ui';

export default function Layout() {
  return <SafeAreaProvider><ConnectionProvider><StatusBar style="dark" />
    <Tabs screenOptions={{ headerShown: false, tabBarActiveTintColor: colors.blue, tabBarInactiveTintColor: colors.muted, tabBarStyle: { borderTopColor: colors.line, paddingTop: 7 }, tabBarLabelStyle: { fontSize: 11, fontWeight: '600' } }}>
      <Tabs.Screen name="index" options={{ title: '품질 검수', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 22 }}>◎</Text> }} />
      <Tabs.Screen name="location" options={{ title: '결함 위치', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 22 }}>⌖</Text> }} />
      <Tabs.Screen name="settings" options={{ title: '연결 설정', tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 22 }}>☷</Text> }} />
    </Tabs>
  </ConnectionProvider></SafeAreaProvider>;
}
