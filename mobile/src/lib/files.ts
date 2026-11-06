import { File, Paths } from 'expo-file-system';
import { fetch as expoFetch } from 'expo/fetch';
import * as ImagePicker from 'expo-image-picker';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import * as DocumentPicker from 'expo-document-picker';
import * as Sharing from 'expo-sharing';
import { Image, Platform } from 'react-native';
import { jsonResponse } from './api';
import type { Photo } from './types';

function dimensions(uri: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => Image.getSize(uri, (width, height) => resolve({ width, height }), reject));
}

export async function pickGallery(): Promise<Photo | null> {
  const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], allowsEditing: false, quality: 1 });
  if (result.canceled) return null;
  const asset = result.assets[0];
  const mime = asset.mimeType || (/\.png$/i.test(asset.fileName || '') ? 'image/png' : 'image/jpeg');
  if (!['image/jpeg', 'image/png'].includes(mime)) {
    const context = ImageManipulator.manipulate(asset.uri);
    const image = await context.renderAsync();
    const converted = await image.saveAsync({ compress: 1, format: SaveFormat.JPEG });
    return { ...converted, name: 'selected_photo.jpg', mime: 'image/jpeg', size: Platform.OS === 'web' ? undefined : new File(converted.uri).size };
  }
  return { uri: asset.uri, name: asset.fileName || `selected_photo.${mime === 'image/png' ? 'png' : 'jpg'}`, mime, width: asset.width, height: asset.height, size: asset.fileSize };
}

export async function pickDocument(): Promise<Photo | null> {
  const result = await DocumentPicker.getDocumentAsync({ type: ['image/jpeg', 'image/png'], copyToCacheDirectory: true, multiple: false });
  if (result.canceled) return null;
  const asset = result.assets[0];
  return { uri: asset.uri, name: asset.name, mime: asset.mimeType || (/\.png$/i.test(asset.name) ? 'image/png' : 'image/jpeg'), size: asset.size, ...await dimensions(asset.uri) };
}

export async function samplePhoto(baseURL: string, path: string, name: string): Promise<Photo> {
  const url = baseURL + path;
  if (Platform.OS === 'web') {
    const response = await fetch(url);
    if (!response.ok) throw new Error('예시 사진을 불러올 수 없습니다.');
    const blob = await response.blob();
    const uri = URL.createObjectURL(blob);
    return { uri, name, mime: blob.type || 'image/png', size: blob.size, ...await dimensions(uri) };
  }
  const file = await File.downloadFileAsync(url, new File(Paths.cache, `${Date.now()}_${name}`));
  return { uri: file.uri, name, mime: /\.jpe?g$/i.test(name) ? 'image/jpeg' : 'image/png', size: file.size, ...await dimensions(file.uri) };
}

export function validatePhoto(photo: Photo, location: boolean): void {
  if (photo.size && photo.size > 30 * 1024 * 1024) throw new Error('사진은 30MB 이하로 선택해 주세요.');
  if (photo.width * photo.height > 25_000_000) throw new Error('사진은 2,500만 픽셀 이하로 선택해 주세요.');
  if (location && (photo.width !== 1600 || photo.height !== 256)) throw new Error('위치 분석은 1600×256 원본만 지원합니다. 예시 사진이나 원본 JPG/PNG를 선택해 주세요.');
}

export async function upload<T>(url: string, photo: Photo, signal: AbortSignal): Promise<T> {
  const body = Platform.OS === 'web' ? await (await fetch(photo.uri, { signal })).blob() : new File(photo.uri);
  return jsonResponse<T>(await expoFetch(url, { method: 'POST', headers: { 'Content-Type': photo.mime }, body, signal }));
}

export async function shareResult(dataURL: string, name: string): Promise<void> {
  const match = /^data:([^;,]+)(?:;[^,]*)?;base64,(.*)$/s.exec(dataURL);
  if (!match) throw new Error('저장할 결과 형식이 올바르지 않습니다.');
  if (Platform.OS === 'web') {
    const response = await fetch(dataURL);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url; link.download = name; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return;
  }
  if (!await Sharing.isAvailableAsync()) throw new Error('이 기기에서 파일 공유를 사용할 수 없습니다.');
  const file = new File(Paths.cache, `${Date.now()}_${name}`);
  await file.write(match[2], { encoding: 'base64' });
  await Sharing.shareAsync(file.uri, { mimeType: match[1], UTI: name.endsWith('.csv') ? 'public.comma-separated-values-text' : 'public.png', dialogTitle: name });
}
