import {BarChart3, FileBarChart, Gauge, ScanLine, ShieldCheck, SlidersHorizontal} from 'lucide-react'

export type WorkspacePage='overview'|'eda'|'preprocess'|'inspection'|'localize'|'performance'

export const workspacePages=[
  {id:'overview' as WorkspacePage,label:'프로젝트 소개',helper:'시스템 개요',icon:Gauge},
  {id:'eda' as WorkspacePage,label:'데이터 EDA',helper:'데이터 품질과 분포',icon:BarChart3},
  {id:'preprocess' as WorkspacePage,label:'이미지 전처리 체험',helper:'파이프라인 미리보기',icon:SlidersHorizontal},
  {id:'inspection' as WorkspacePage,label:'자동 품질 검수',helper:'RF · ResNet18 실제 추론',icon:ShieldCheck},
  {id:'localize' as WorkspacePage,label:'정밀 결함 위치',helper:'U-Net 분할과 좌표',icon:ScanLine},
  {id:'performance' as WorkspacePage,label:'모델 성능 비교',helper:'평가 리포트',icon:FileBarChart},
]

export const workspaceMeta:Record<WorkspacePage,{eyebrow:string;title:string;description:string}>={
  overview:{eyebrow:'STEELVISION WORKSPACE',title:'철강 표면 품질을 한눈에',description:'데이터 탐색부터 결함 판정, 위치 분석까지 하나의 검수 작업 공간으로 연결합니다.'},
  eda:{eyebrow:'DATA EXPLORER',title:'데이터 EDA',description:'실제 데이터 구성과 결함 주석, 원본 이미지의 출처를 확인합니다.'},
  preprocess:{eyebrow:'PREPROCESSING LAB',title:'이미지 전처리 체험',description:'모델 추론과 분리된 샌드박스에서 실제 영상 처리 단계를 비교합니다.'},
  inspection:{eyebrow:'QUALITY INSPECTION',title:'자동 품질 검수',description:'이미지를 업로드하고 연결된 분류 모델로 정상·불량을 판정합니다.'},
  localize:{eyebrow:'DEFECT LOCALIZATION',title:'정밀 결함 위치',description:'U-Net의 실제 예측 마스크와 위치 좌표를 원본 기준으로 확인합니다.'},
  performance:{eyebrow:'MODEL REPORT',title:'모델 성능 비교',description:'현재 PC의 평가와 노트북에 저장된 후속 비교를 구분해 살펴봅니다.'},
}

export const workspaceGroups=[
  {title:'프로젝트',number:'01',links:[['프로젝트 소개','overview'],['처리 흐름과 모델','overview']]},
  {title:'데이터',number:'02',links:[['데이터 EDA','eda'],['원본 사진·정답 예시','eda'],['이미지 전처리 체험','preprocess']]},
  {title:'품질 검수',number:'03',links:[['자동 품질 검수','inspection'],['정밀 결함 위치','localize']]},
  {title:'모델 리포트',number:'04',links:[['모델 성능 비교','performance'],['현재 PC·노트북 결과','performance']]},
]

export const workspaceHref=(page:string)=>`#/dashboard/${page}`
