# SteelVision ML Mobile

Expo·React Native 앱에서 기존 Python 모델로 사진의 품질과 결함 위치를 분석합니다.

프로젝트 최상위의 **`Start_SteelVision_Expo.cmd`**를 실행하고, 같은 Wi-Fi의 아이폰에서 QR을 스캔하세요. 설치·연결·파일 공유 안내는 [모바일 실행 문서](../05_모바일_Expo_실행방법.md)에 있습니다.

```powershell
npm.cmd ci
npm.cmd run login
npm.cmd start
```

`npm.cmd run web`은 브라우저 미리보기, `npm.cmd run typecheck`와 `npm.cmd run lint`는 코드 검사를 실행합니다. 사용자 아이폰의 Expo Go와 같은 SDK 58 호환 패키지와 lockfile을 사용합니다.

모바일 API 기본 포트는 8504이며, 모델 가중치·전처리·임계값은 기존 프로젝트의 값을 사용합니다. 모델 추론은 PC에서 수행합니다.
