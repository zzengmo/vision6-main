# SteelVision ML · 아이폰 Expo 실행 방법

## 실행

1. 아이폰에 **Expo Go**를 설치하고 Expo 계정으로 로그인합니다.
2. PC에서도 **같은 Expo 계정**으로 로그인합니다. 아래 ‘최초 로그인’을 참고하세요.
3. PC와 아이폰을 **같은 Wi-Fi**에 연결합니다.
4. 프로젝트 최상위 **`Start_SteelVision_Expo.cmd`를 더블클릭**합니다.
5. PC 실행 창의 **QR을 아이폰 기본 카메라로 스캔 → Expo Go에서 열기**를 선택합니다.
6. 로컬 네트워크 접근 요청은 허용합니다. 앱 상단에 **서버 연결됨**이 표시되면 준비 완료입니다.

실행 창은 사용 중 열어 둡니다. `Ctrl+C`로 종료하면 이 실행 파일이 시작한 API와 Expo 서버가 함께 종료됩니다. 기존 PC 웹의 `Start_SteelVision.cmd`도 계속 사용할 수 있습니다. 최초 패키지 설치에는 인터넷이 필요합니다.

## 최초 로그인 · 아이폰과 PC 계정 맞추기

실제 아이폰에서는 **Expo Go와 PC의 Expo CLI가 같은 계정에 로그인**되어 있어야 합니다. 프로젝트 최상위에서 PowerShell을 열고 실행합니다.

```powershell
cd .\mobile
npm.cmd run login
npx.cmd expo whoami
```

브라우저에서 아이폰 Expo Go와 같은 계정으로 로그인합니다. `whoami`에 표시된 이름이 아이폰 계정과 같으면 Expo Go의 **Try Again**을 누르거나 QR을 다시 스캔합니다. 다른 계정에서 전환했다면 실행 창을 `Ctrl+C`로 종료하고 다시 실행합니다. [Expo 공식 로그인 안내](https://docs.expo.dev/troubleshooting/expo-go-sign-in-required/)

`npm.cmd run login`은 Windows에서 로그인 URL이 잘리는 문제를 피하도록 Chrome 또는 Edge를 직접 엽니다. 기본 Expo 명령은 `npx.cmd expo login --browser`입니다. 비밀번호·인증 코드는 Expo 공식 로그인 화면에서 입력합니다.

## 사진 검사와 결과 저장

**품질 검수**에서 사진 보관함·파일 앱의 사진을 선택하고 모델·운영점을 정한 뒤 **품질 검수 실행**을 누릅니다. 기본 모델은 **ResNet18 + 좌우반전 TTA**입니다. 직접 사진이 없어도 **정상 예시·불량 예시·제공 사진** 버튼을 사용할 수 있습니다.

- JPG·PNG는 **30MB·2,500만 픽셀 이하**를 사용합니다.
- 보관함의 HEIC 사진은 크기를 유지하며 JPG로 변환합니다.
- **결함 위치** 탭의 U-Net은 **1600×256 원본**만 허용합니다. 일반 아이폰 사진을 임의로 늘리거나 잘라 넣지 않습니다. 예시 사진은 이 규격으로 준비되어 있습니다.
- 결과 사진을 누르면 확대할 수 있습니다. 아이폰 확대 화면에서 두 손가락으로 확대합니다.
- **PNG·CSV 공유 → 파일에 저장**으로 저장하거나 다른 앱으로 전달합니다. 공유는 이미 계산한 결과를 사용하며 재추론하지 않습니다.

모델은 PC에서 실행합니다. 검사 중 PC 서버가 켜져 있어야 하며 휴대폰에서 모델을 학습하지 않습니다. 품질 판정과 U-Net 위치 분석은 별도 실행입니다.

## 연결이 안 될 때

앱 하단 **연결 설정**에서 PC 실행 창에 표시된 주소를 입력합니다.

```text
http://PC의_Wi-Fi_IP:8504
예: http://192.168.0.10:8504
```

휴대폰의 `127.0.0.1`은 휴대폰 자신입니다. PC의 Wi-Fi IP를 사용하세요. VPN·가상 어댑터가 있으면 자동 주소 대신 실제 Wi-Fi 어댑터 주소를 입력해야 할 수 있습니다.

- 아이폰 **설정 → 개인정보 보호 및 보안 → 로컬 네트워크**에서 Expo Go 접근을 확인합니다.
- Windows 방화벽 안내가 뜨면 개인 네트워크에서 Node.js·Python 통신을 허용합니다.
- 공유기의 게스트망·기기 격리 설정은 같은 Wi-Fi의 기기 접속도 막을 수 있습니다.
- 앱에 입력한 서버 주소를 아이폰 Safari에서 열어 PC 서버에 접속할 수 있는지 확인할 수 있습니다.
- Expo QR 연결과 Python API 연결은 별개입니다. Expo 터널만 켜도 Python API가 외부에 공개되는 것은 아닙니다.

## 아이폰 Expo Go 호환성

사용자 아이폰에 설치된 **Expo Go SDK 58.0.0**에 맞춰 프로젝트를 **Expo SDK 58**로 구성했습니다. Expo Go와 프로젝트의 SDK 주 버전이 같아야 합니다. [Expo 공식 업그레이드 안내](https://docs.expo.dev/workflow/upgrading-expo-sdk-walkthrough/)

`package-lock.json`에 설치 버전을 고정했습니다. Expo 버전을 바꿀 때는 설치된 Expo Go와 SDK의 호환성을 함께 확인합니다.

이전 SDK 54를 실행했던 창은 `Ctrl+C`로 종료하고 `Start_SteelVision_Expo.cmd --clear`로 다시 실행해 새 QR을 스캔합니다. 휴대폰의 Expo Go도 완전히 닫았다가 다시 여세요.

## 개발 명령과 파일

PowerShell에서는 실행 정책 변경 없이 `npm.cmd`·`npx.cmd`를 사용합니다.

```powershell
cd .\mobile
npm.cmd ci
npm.cmd start
```

PC 브라우저에서 모바일 화면을 확인하려면 `npm.cmd run web`을 실행합니다.

| 파일 | 역할 |
| --- | --- |
| `mobile/src/app/index.tsx` | 사진 선택·모델 선택·품질 검수 |
| `mobile/src/app/location.tsx` | U-Net 마스크·원본 좌표·결과 공유 |
| `mobile/src/app/settings.tsx` | 서버 주소 저장·연결 확인 |
| `mobile/src/lib/files.ts` | 파일 선택·원본 바이트 업로드·PNG/CSV 공유 |
| `mobile/scripts/start.mjs` | 모바일 API와 Expo 시작·종료 |

기본 포트는 **기존 웹 8502 / 모바일 API 8504 / Expo 8081**입니다. 변경하려면 `STEELVISION_API_PORT`, `STEELVISION_EXPO_PORT` 환경 변수를 지정합니다. `EXPO_PUBLIC_API_URL`로 초기 서버 주소를 지정하거나 앱에서 직접 바꿀 수 있습니다.

Python 서버의 기본 실행은 기존처럼 `127.0.0.1`에 한정됩니다. Expo 실행 파일이 모바일 API를 실행할 때에만 `--host 0.0.0.0`으로 같은 네트워크의 휴대폰 접속을 허용합니다. 브라우저 CORS도 지정한 Expo 미리보기 주소에만 적용합니다.
