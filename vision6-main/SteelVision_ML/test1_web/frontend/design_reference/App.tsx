import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react"
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Boxes,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Database,
  Download,
  Eye,
  FileBarChart,
  FileImage,
  Gauge,
  Grid3X3,
  Image as ImageIcon,
  Info,
  Layers3,
  Menu,
  Microscope,
  PanelLeftClose,
  Play,
  ScanLine,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  UploadCloud,
  X,
  Zap,
} from "lucide-react"

type PageId = "overview" | "eda" | "preprocess" | "inspection" | "localize" | "performance"

const pages: {
  id: PageId
  label: string
  helper: string
  icon: typeof Activity
}[] = [
  {
    id: "overview",
    label: "프로젝트 소개",
    helper: "시스템 개요",
    icon: Gauge,
  },
  { id: "eda", label: "데이터 EDA", helper: "데이터 품질", icon: BarChart3 },
  {
    id: "preprocess",
    label: "이미지 전처리 체험",
    helper: "파이프라인 미리보기",
    icon: SlidersHorizontal,
  },
  {
    id: "inspection",
    label: "자동 품질 검수",
    helper: "결함 유무 판정",
    icon: ShieldCheck,
  },
  {
    id: "localize",
    label: "정밀 결함 위치",
    helper: "U-Net · YOLO",
    icon: ScanLine,
  },
  {
    id: "performance",
    label: "모델 성능 비교",
    helper: "평가 리포트",
    icon: FileBarChart,
  },
]

const pageMeta: Record<PageId, {
  eyebrow: string
  title: string
  description: string
}> = {
  overview: {
    eyebrow: "STEELVISION WORKSPACE",
    title: "철강 표면 품질을 한눈에",
    description:
      "데이터 탐색부터 결함 판정, 위치 추적까지 하나의 검수 흐름으로 관리합니다.",
  },
  eda: {
    eyebrow: "DATA EXPLORER",
    title: "데이터 EDA",
    description: "Severstal 학습 데이터의 구성과 결함 분포를 검토합니다.",
  },
  preprocess: {
    eyebrow: "PREPROCESSING LAB",
    title: "이미지 전처리 체험",
    description: "모델 추론과 분리된 샌드박스에서 영상 처리 단계를 비교합니다.",
  },
  inspection: {
    eyebrow: "QUALITY INSPECTION",
    title: "자동 품질 검수",
    description:
      "이미지를 업로드하고 연결된 분류 모델로 결함 유무를 판정합니다.",
  },
  localize: {
    eyebrow: "DEFECT LOCALIZATION",
    title: "정밀 결함 위치",
    description:
      "분할 모델의 결함 마스크와 위치 좌표를 원본 기준으로 확인합니다.",
  },
  performance: {
    eyebrow: "MODEL REPORT",
    title: "모델 성능 비교",
    description:
      "같은 홀드아웃 데이터에서 평가한 모델의 품질과 처리 속도를 비교합니다.",
  },
}

function Card({
  children,
  className = "",
  title,
  action,
}: {
  children: ReactNode
  className?: string
  title?: string
  action?: ReactNode
}) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <div className="card-heading">
          {title && <h3>{title}</h3>}
          {action}
        </div>
      )}
      {children}
    </section>
  )
}

function Pill({
  children,
  tone = "neutral",
}: {
  children: ReactNode
  tone?: "neutral" | "blue" | "green" | "amber"
}) {
  return <span className={`pill pill-${tone}`}>{children}</span>
}

function Metric({
  label,
  value,
  detail,
  icon: Icon,
  tone = "blue",
}: {
  label: string
  value: string
  detail: string
  icon: typeof Activity
  tone?: "blue" | "green" | "amber" | "slate"
}) {
  return (
    <Card className="metric-card">
      <div className={`metric-icon metric-${tone}`}>
        <Icon size={19} />
      </div>
      <div>
        <p className="metric-label">{label}</p>
        <strong className="metric-value">{value}</strong>
        <p className="metric-detail">{detail}</p>
      </div>
    </Card>
  )
}

function InfoBanner({
  children,
  tone = "blue",
}: {
  children: ReactNode
  tone?: "blue" | "amber" | "slate"
}) {
  const Icon =
    tone === "amber" ? AlertTriangle : tone === "slate" ? Info : CheckCircle2
  return (
    <div className={`info-banner info-${tone}`}>
      <Icon size={18} />
      <div>{children}</div>
    </div>
  )
}

function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean
  onChange: () => void
  label: string
  hint?: string
}) {
  return (
    <button className="toggle-row" onClick={onChange} type="button">
      <span>
        <strong>{label}</strong>
        {hint && <small>{hint}</small>}
      </span>
      <span className={`toggle ${checked ? "is-on" : ""}`}>
        <span />
      </span>
    </button>
  )
}

function Bar({
  label,
  value,
  max,
  color = "blue",
  suffix = "",
}: {
  label: string
  value: number
  max: number
  color?: string
  suffix?: string
}) {
  return (
    <div className="bar-row">
      <div className="bar-label">
        <span>{label}</span>
        <strong>
          {value.toLocaleString()}
          {suffix}
        </strong>
      </div>
      <div className="bar-track">
        <span
          className={`bar-fill fill-${color}`}
          style={{ width: `${Math.max(3, (value / max) * 100)}%` }}
        />
      </div>
    </div>
  )
}

function ScoreRing({ value, label }: { value: number label: string }) {
  const deg = value * 3.6
  return (
    <div className="score-wrap">
      <div
        className="score-ring"
        style={{
          background: `conic-gradient(#2563eb ${deg}deg, #e8eef7 0deg)`,
        }}
      >
        <div>
          <strong>{value}</strong>
          <span>%</span>
        </div>
      </div>
      <p>{label}</p>
    </div>
  )
}

function UploadArea({
  file,
  onFile,
  compact = false,
}: {
  file: File | null
  onFile: (file: File | null) => void
  compact?: boolean
}) {
  const id = useMemo(() => `upload-${Math.random().toString(36).slice(2)}`, [])
  const handle = (event: ChangeEvent<HTMLInputElement>) => {
    const next = event.target.files?.[0] ?? null
    if (next && next.size > 30 * 1024 * 1024) {
      alert("파일 크기는 30MB를 초과할 수 없습니다.")
      return
    }
    onFile(next)
  }
  return (
    <label
      className={`upload-area ${compact ? "compact" : ""} ${
        file ? "has-file" : ""
      }`}
      htmlFor={id}
    >
      <input
        id={id}
        type="file"
        accept="image/jpeg,image/png"
        onChange={handle}
      />
      <div className="upload-icon">
        {file ? <FileImage size={24} /> : <UploadCloud size={25} />}
      </div>
      {file ? (
        <div className="upload-copy">
          <strong>{file.name}</strong>
          <span>{(file.size / 1024 / 1024).toFixed(2)} MB · 클릭하여 교체</span>
        </div>
      ) : (
        <div className="upload-copy">
          <strong>검수할 이미지를 끌어놓거나 선택하세요</strong>
          <span>JPG, PNG · 최대 30MB · 2,500만 픽셀 이하</span>
        </div>
      )}
      <span className="outline-button">파일 선택</span>
    </label>
  )
}

function OverviewPage({ navigate }: { navigate: (page: PageId) => void }) {
  return (
    <>
      <div className="overview-hero">
        <div className="hero-copy">
          <Pill tone="green">
            <span className="status-dot" /> 시스템 준비됨
          </Pill>
          <h2>
            현장의 표면 결함,
            <br />
            <em>더 빠르고 정확하게.</em>
          </h2>
          <p>
            SteelVision ML은 철강 이미지의 이상 유무를 판정하고 결함 위치를
            시각화하는 품질 검수 워크스페이스입니다.
          </p>
          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={() => navigate("inspection")}
            >
              <Play size={17} fill="currentColor" /> 검수 시작하기
            </button>
            <button
              className="secondary-button"
              onClick={() => navigate("performance")}
            >
              성능 리포트 보기 <ChevronRight size={17} />
            </button>
          </div>
        </div>
        <div className="hero-visual" aria-label="철강 표면 검사 과정 시각화">
          <div className="steel-panel">
            <span className="steel-line line-1" />
            <span className="steel-line line-2" />
            <span className="steel-line line-3" />
            <span className="scan-beam" />
            <span className="defect-mark defect-a">
              <i>01</i>
            </span>
            <span className="defect-mark defect-b">
              <i>02</i>
            </span>
          </div>
          <div className="visual-float float-top">
            <Zap size={15} />
            <span>
              <small>평균 처리 시간</small>
              <strong>6.92 ms</strong>
            </span>
          </div>
          <div className="visual-float float-bottom">
            <CheckCircle2 size={16} />
            <span>
              <small>검수 상태</small>
              <strong>분류 모델 준비</strong>
            </span>
          </div>
        </div>
      </div>

      <div className="metric-grid four">
        <Metric
          label="학습 이미지"
          value="12,568"
          detail="공식 Severstal 데이터"
          icon={Database}
          tone="blue"
        />
        <Metric
          label="결함 재현율"
          value="95.6%"
          detail="홀드아웃 2,514장"
          icon={ShieldCheck}
          tone="green"
        />
        <Metric
          label="ROC-AUC"
          value="0.949"
          detail="이미지 단위 기준선"
          icon={Activity}
          tone="slate"
        />
        <Metric
          label="결함 클래스"
          value="4종"
          detail="물리 명칭 미확인"
          icon={Layers3}
          tone="amber"
        />
      </div>

      <div className="two-col overview-lower">
        <Card title="검수 워크플로" action={<Pill tone="blue">운영 흐름</Pill>}>
          <div className="workflow">
            {[
              [UploadCloud, "01", "이미지 입력", "JPG · PNG 원본"],
              [Sparkles, "02", "특징 추출", "200 × 32 · 39개 특징"],
              [Microscope, "03", "모델 분석", "RandomForest 기준선"],
              [ShieldCheck, "04", "품질 판정", "정상 · 결함 분류"],
            ].map(([Icon, no, title, text], index) => {
              const FlowIcon = Icon as typeof Activity
              return (
                <div className="workflow-step" key={String(no)}>
                  <span className="step-no">{String(no)}</span>
                  <div className="step-icon">
                    <FlowIcon size={21} />
                  </div>
                  <div>
                    <strong>{String(title)}</strong>
                    <small>{String(text)}</small>
                  </div>
                  {index < 3 && (
                    <ChevronRight className="flow-arrow" size={18} />
                  )}
                </div>
              )
            })}
          </div>
        </Card>
        <Card
          title="현재 실험 정보"
          action={
            <button className="text-button">
              실행 보고서 <ChevronRight size={15} />
            </button>
          }
        >
          <div className="experiment-info">
            <div>
              <span>실험 ID</span>
              <strong>test1 · first-baseline</strong>
            </div>
            <div>
              <span>분류 모델</span>
              <strong>RandomForest · 180 trees</strong>
            </div>
            <div>
              <span>위치 모델</span>
              <strong>U-Net 선택 · YOLO 비교</strong>
            </div>
            <div>
              <span>데이터 분할</span>
              <strong>Train 60 · Val 20 · Test 20%</strong>
            </div>
          </div>
          <InfoBanner tone="slate">
            이 화면의 지표는 첨부된 노트북에 저장된 실험 결과이며, 생산 라인
            성능을 보증하지 않습니다.
          </InfoBanner>
        </Card>
      </div>
    </>
  )
}

function EdaPage() {
  const classes = [
    { name: "Class 1", value: 897, color: "blue" },
    { name: "Class 2", value: 247, color: "cyan" },
    { name: "Class 3", value: 5150, color: "indigo" },
    { name: "Class 4", value: 801, color: "amber" },
  ]
  return (
    <>
      <div className="metric-grid three">
        <Metric
          label="전체 이미지"
          value="12,568"
          detail="1600 × 256 px"
          icon={Database}
        />
        <Metric
          label="정상 이미지"
          value="5,902"
          detail="전체의 47.0%"
          icon={CheckCircle2}
          tone="green"
        />
        <Metric
          label="결함 이미지"
          value="6,666"
          detail="전체의 53.0%"
          icon={AlertTriangle}
          tone="amber"
        />
      </div>
      <InfoBanner tone="blue">
        <strong>데이터 기준</strong> · train.csv에 결함 행이 없는 이미지는
        정상으로 분류했습니다. 클래스 1–4의 물리적 명칭은 원본 CSV만으로 확인할
        수 없습니다.
      </InfoBanner>
      <div className="two-col eda-grid">
        <Card
          title="정상 · 결함 분포"
          action={<Pill tone="neutral">이미지 단위</Pill>}
        >
          <div className="donut-layout">
            <div className="donut">
              <div>
                <strong>12,568</strong>
                <span>전체 이미지</span>
              </div>
            </div>
            <div className="legend-list">
              <div>
                <span>
                  <i className="legend defect" />
                  결함
                </span>
                <strong>
                  6,666 <small>53.0%</small>
                </strong>
              </div>
              <div>
                <span>
                  <i className="legend normal" />
                  정상
                </span>
                <strong>
                  5,902 <small>47.0%</small>
                </strong>
              </div>
            </div>
          </div>
        </Card>
        <Card
          title="클래스별 라벨 행"
          action={<CircleHelp size={17} className="muted-icon" />}
        >
          <div className="bars">
            {classes.map((item) => (
              <Bar
                key={item.name}
                label={item.name}
                value={item.value}
                max={5150}
                color={item.color}
              />
            ))}
          </div>
          <p className="chart-note">
            한 이미지에 여러 결함 클래스가 존재할 수 있어 합계는 결함 이미지
            수와 다릅니다.
          </p>
        </Card>
      </div>
      <Card
        title="데이터 분할 구성"
        action={
          <Pill tone="green">
            <Check size={13} /> 이미지 ID 중복 없음
          </Pill>
        }
      >
        <div className="split-grid">
          {[
            ["학습", "7,540", "3,999", "3,541", 60],
            ["검증", "2,514", "1,334", "1,180", 20],
            ["홀드아웃 시험", "2,514", "1,333", "1,181", 20],
          ].map(([name, total, defect, normal, percent]) => (
            <div className="split-item" key={String(name)}>
              <div className="split-top">
                <strong>{name}</strong>
                <span>{percent}%</span>
              </div>
              <div className="split-total">
                {total}
                <small>장</small>
              </div>
              <div className="split-track">
                <i style={{ width: "53%" }} />
              </div>
              <div className="split-detail">
                <span>결함 {defect}</span>
                <span>정상 {normal}</span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </>
  )
}

function ImagePreview({
  url,
  filtered = false,
}: {
  url: string | null
  filtered?: boolean
}) {
  if (!url) {
    return (
      <div className="empty-image">
        <ImageIcon size={30} />
        <span>이미지를 업로드하면 여기에 표시됩니다</span>
      </div>
    )
  }
  return (
    <img
      className={filtered ? "filtered-image" : ""}
      src={url}
      alt={filtered ? "전처리 결과" : "업로드 원본"}
    />
  )
}

function PreprocessPage() {
  const [file, setFile] = useState<File | null>(null)
  const [settings, setSettings] = useState({
    clahe: true,
    bilateral: true,
    sharpen: false,
    opening: false,
    canny: false,
    grid: false,
  })
  const url = useObjectUrl(file)
  const toggle = (key: keyof typeof settings) =>
    setSettings((value) => ({ ...value, [key]: !value[key] }))
  return (
    <>
      <InfoBanner tone="amber">
        <strong>체험용 샌드박스</strong> · 아래 옵션은 시각적 비교 전용이며 실제
        모델의 학습·추론 전처리를 변경하지 않습니다.
      </InfoBanner>
      <div className="lab-layout">
        <Card className="settings-card" title="처리 옵션">
          <UploadArea file={file} onFile={setFile} compact />
          <div className="setting-group">
            <p className="group-label">화질 개선</p>
            <Toggle
              checked={settings.clahe}
              onChange={() => toggle("clahe")}
              label="CLAHE"
              hint="국부 대비 향상"
            />
            <Toggle
              checked={settings.bilateral}
              onChange={() => toggle("bilateral")}
              label="Bilateral Filter"
              hint="에지 보존 노이즈 제거"
            />
            <Toggle
              checked={settings.sharpen}
              onChange={() => toggle("sharpen")}
              label="Adaptive Sharpen"
              hint="미세 구조 선명화"
            />
            <Toggle
              checked={settings.opening}
              onChange={() => toggle("opening")}
              label="Opening"
              hint="작은 밝은 노이즈 제거"
            />
          </div>
          <div className="setting-group">
            <p className="group-label">오버레이</p>
            <Toggle
              checked={settings.canny}
              onChange={() => toggle("canny")}
              label="Canny Edge"
            />
            <Toggle
              checked={settings.grid}
              onChange={() => toggle("grid")}
              label="ROI Grid"
            />
          </div>
          <button className="secondary-button wide" disabled={!file}>
            <Download size={16} /> 전처리 PNG 다운로드
          </button>
        </Card>
        <div className="lab-content">
          <Card
            title="원본 · 전처리 비교"
            action={<Pill tone="blue">실시간 미리보기</Pill>}
          >
            <div className="compare-grid">
              <div className="image-panel">
                <div className="image-label">
                  <span>원본</span>
                  <small>Original</small>
                </div>
                <div className="preview-box">
                  <ImagePreview url={url} />
                </div>
              </div>
              <div className="compare-arrow">
                <ChevronRight size={20} />
              </div>
              <div className="image-panel">
                <div className="image-label">
                  <span>처리 결과</span>
                  <small>
                    {Object.values(settings).filter(Boolean).length}개 옵션 적용
                  </small>
                </div>
                <div
                  className={`preview-box ${settings.canny ? "edge-on" : ""} ${
                    settings.grid ? "grid-on" : ""
                  }`}
                >
                  <ImagePreview url={url} filtered />
                </div>
              </div>
            </div>
          </Card>
          <div className="two-col mini">
            <Card title="영상 측정값">
              <div className="measure-list">
                <div>
                  <span>평균 밝기</span>
                  <strong>{file ? "128.4" : "—"}</strong>
                </div>
                <div>
                  <span>대비 표준편차</span>
                  <strong>{file ? "42.7" : "—"}</strong>
                </div>
                <div>
                  <span>에지 밀도</span>
                  <strong>{file ? "8.31%" : "—"}</strong>
                </div>
              </div>
            </Card>
            <Card title="처리 단계">
              <div className="stage-row">
                {["입력", "대비", "필터", "강조"].map((text, i) => (
                  <div key={text}>
                    <span className={i < 3 && file ? "done" : ""}>
                      {i < 3 && file ? <Check size={13} /> : i + 1}
                    </span>
                    <small>{text}</small>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </>
  )
}

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!file) {
      setUrl(null)
      return
    }
    const next = URL.createObjectURL(file)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [file])
  return url
}

function InspectionPage() {
  const [file, setFile] = useState<File | null>(null)
  const [running, setRunning] = useState(false)
  const [attempted, setAttempted] = useState(false)
  const url = useObjectUrl(file)
  const run = () => {
    if (!file) return
    setRunning(true)
    setAttempted(false)
    window.setTimeout(() => {
      setRunning(false)
      setAttempted(true)
    }, 900)
  }
  return (
    <>
      <InfoBanner tone="slate">
        <strong>연결 상태</strong> · 저장된 평가 메타데이터는 확인했지만
        브라우저에서 호출할 실제 Python 추론 API는 아직 연결되지 않았습니다.
      </InfoBanner>
      <div className="inspection-layout">
        <div>
          <Card title="1. 검사 이미지">
            <UploadArea
              file={file}
              onFile={(next) => {
                setFile(next)
                setAttempted(false)
              }}
            />
            {url && (
              <div className="uploaded-preview">
                <img src={url} alt="검사 이미지 미리보기" />
                <button onClick={() => setFile(null)}>
                  <X size={15} />
                </button>
              </div>
            )}
          </Card>
          <Card title="2. 검사 설정" className="top-gap">
            <label className="field-label">분류 모델</label>
            <div className="select-like">
              <span>
                <Boxes size={17} /> RandomForest · 200×32
              </span>
              <ChevronDown size={17} />
            </div>
            <div className="model-details">
              <div>
                <span>판정 단위</span>
                <strong>이미지</strong>
              </div>
              <div>
                <span>결함 기준</span>
                <strong>Score ≥ 0.381</strong>
              </div>
              <div>
                <span>입력 특징</span>
                <strong>39개 통계 특징</strong>
              </div>
            </div>
            <label className="field-label disabled-label">
              최소 불량 패치 수 <Pill tone="neutral">미지원</Pill>
            </label>
            <div className="disabled-control">
              패치 집계 기능을 지원하지 않는 모델입니다.
            </div>
            <button
              className="primary-button wide run-button"
              onClick={run}
              disabled={!file || running}
            >
              {running ? (
                <>
                  <span className="spinner" /> 연결 확인 중...
                </>
              ) : (
                <>
                  <Play size={17} fill="currentColor" /> 품질 검사 실행
                </>
              )}
            </button>
          </Card>
        </div>
        <Card
          className="result-card"
          title="검사 결과"
          action={
            <Pill tone={attempted ? "amber" : "neutral"}>
              {attempted ? "연결 필요" : "실행 대기"}
            </Pill>
          }
        >
          {!attempted ? (
            <div className="empty-result">
              <div className="result-orbit">
                <ScanLine size={34} />
              </div>
              <strong>
                {running
                  ? "추론 연결 상태를 확인하고 있습니다"
                  : "검사 결과가 여기에 표시됩니다"}
              </strong>
              <p>
                {file
                  ? "왼쪽의 ‘품질 검사 실행’ 버튼을 눌러주세요."
                  : "이미지를 업로드하고 모델을 선택해 주세요."}
              </p>
            </div>
          ) : (
            <div className="connection-result">
              <div className="connection-icon">
                <AlertTriangle size={25} />
              </div>
              <h3>실시간 추론 어댑터가 필요합니다</h3>
              <p>
                가짜 판정 결과를 표시하지 않았습니다. Python 모델 API를 연결하면
                판정, 점수, 처리 시간 및 CSV 다운로드가 이 영역에 활성화됩니다.
              </p>
              <div className="required-fields">
                <div>
                  <Check size={15} />
                  <span>실험 메타데이터</span>
                  <strong>확인됨</strong>
                </div>
                <div>
                  <X size={15} />
                  <span>가중치 경로</span>
                  <strong>로컬 절대경로</strong>
                </div>
                <div>
                  <X size={15} />
                  <span>추론 엔드포인트</span>
                  <strong>미연결</strong>
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>
    </>
  )
}

function LocalizePage() {
  const [file, setFile] = useState<File | null>(null)
  const [threshold, setThreshold] = useState(55)
  const url = useObjectUrl(file)
  return (
    <>
      <div className="model-strip">
        <div className="model-title">
          <div className="model-logo">
            <ScanLine size={21} />
          </div>
          <div>
            <Pill tone="green">노트북 선택 모델</Pill>
            <h3>U-Net Binary Segmentation</h3>
            <p>400×256 패치 4개 · 회색조 1채널 · BCE + Dice</p>
          </div>
        </div>
        <div className="model-strip-metrics">
          <div>
            <span>Test Dice</span>
            <strong>0.478</strong>
          </div>
          <div>
            <span>Test IoU</span>
            <strong>0.314</strong>
          </div>
          <div>
            <span>Recall</span>
            <strong>96.9%</strong>
          </div>
          <div>
            <span>Median</span>
            <strong>36.5ms</strong>
          </div>
        </div>
      </div>
      <InfoBanner tone="amber">
        위 지표는 120장 홀드아웃의 저장된 실험 결과입니다. 가중치 파일은
        첨부되지 않아 새 이미지 위치 예측은 아직 실행할 수 없습니다.
      </InfoBanner>
      <div className="localize-layout">
        <Card className="local-controls" title="위치 찾기 설정">
          <UploadArea file={file} onFile={setFile} compact />
          <div className="slider-heading">
            <span>검출 임계값</span>
            <strong>{(threshold / 100).toFixed(2)}</strong>
          </div>
          <input
            className="range"
            type="range"
            min="10"
            max="90"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <div className="range-labels">
            <span>민감하게</span>
            <span>정밀하게</span>
          </div>
          <div className="option-box">
            <div>
              <Layers3 size={17} />
              <span>
                <strong>예측 마스크</strong>
                <small>U-Net 지원</small>
              </span>
            </div>
            <span className="check-box checked">
              <Check size={13} />
            </span>
          </div>
          <div className="option-box">
            <div>
              <Boxes size={17} />
              <span>
                <strong>연결 영역 박스</strong>
                <small>최대 2개 · sum 순위</small>
              </span>
            </div>
            <span className="check-box checked">
              <Check size={13} />
            </span>
          </div>
          <button className="primary-button wide" disabled>
            <Search size={17} /> 가중치 연결 후 위치 찾기
          </button>
        </Card>
        <Card
          title="원본 · 예측 결과"
          action={
            <span className="legend-caption">
              <i /> 결함 예측 영역
            </span>
          }
        >
          <div className="local-compare">
            <div>
              <div className="image-label">
                <span>원본 이미지</span>
                <small>1600 × 256 기준</small>
              </div>
              <div className="wide-preview">
                <ImagePreview url={url} />
              </div>
            </div>
            <div>
              <div className="image-label">
                <span>예측 결과</span>
                <small>원본 좌표로 복원</small>
              </div>
              <div className="wide-preview prediction-placeholder">
                <ImagePreview url={url} />
                {url && (
                  <div className="not-run-overlay">
                    <Eye size={20} />
                    <span>모델 미연결</span>
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="download-row">
            {["결과 PNG", "마스크 PNG", "좌표 CSV"].map((item) => (
              <button disabled key={item}>
                <Download size={15} /> {item}
              </button>
            ))}
          </div>
        </Card>
      </div>
    </>
  )
}

function PerformancePage() {
  const [tab, setTab] = useState<"classification" | "segmentation">(
    "classification",
  )
  return (
    <>
      <div className="report-tabs">
        <button
          className={tab === "classification" ? "active" : ""}
          onClick={() => setTab("classification")}
        >
          <ShieldCheck size={17} /> 이미지 분류
        </button>
        <button
          className={tab === "segmentation" ? "active" : ""}
          onClick={() => setTab("segmentation")}
        >
          <ScanLine size={17} /> 위치 분할
        </button>
      </div>
      {tab === "classification" ? (
        <>
          <div className="metric-grid four">
            <Metric
              label="Average Precision"
              value="0.950"
              detail="홀드아웃 시험"
              icon={Activity}
              tone="blue"
            />
            <Metric
              label="ROC-AUC"
              value="0.949"
              detail="점수 기반"
              icon={BarChart3}
              tone="slate"
            />
            <Metric
              label="Recall"
              value="95.6%"
              detail="FN 59장"
              icon={ShieldCheck}
              tone="green"
            />
            <Metric
              label="Precision"
              value="82.4%"
              detail="FP 272장"
              icon={Gauge}
              tone="amber"
            />
          </div>
          <div className="two-col performance-grid">
            <Card
              title="홀드아웃 혼동 행렬"
              action={<Pill tone="neutral">Threshold 0.381</Pill>}
            >
              <div className="matrix-wrap">
                <div className="matrix-axis y">실제 클래스</div>
                <div className="matrix">
                  <div className="matrix-cell light">
                    <span>정상 → 정상</span>
                    <strong>909</strong>
                    <small>TN</small>
                  </div>
                  <div className="matrix-cell pale-red">
                    <span>정상 → 결함</span>
                    <strong>272</strong>
                    <small>FP</small>
                  </div>
                  <div className="matrix-cell pale-red">
                    <span>결함 → 정상</span>
                    <strong>59</strong>
                    <small>FN</small>
                  </div>
                  <div className="matrix-cell deep">
                    <span>결함 → 결함</span>
                    <strong>1,274</strong>
                    <small>TP</small>
                  </div>
                </div>
                <div className="matrix-axis x">예측 클래스</div>
              </div>
            </Card>
            <Card
              title="클래스별 결함 경보 재현율"
              action={<Pill tone="blue">복수 클래스 포함</Pill>}
            >
              <div className="score-grid">
                <ScoreRing value={99.5} label="Class 1" />
                <ScoreRing value={100} label="Class 2" />
                <ScoreRing value={95.0} label="Class 3" />
                <ScoreRing value={94.3} label="Class 4" />
              </div>
              <InfoBanner tone="slate">
                이미지 단위 경보 재현율이며 픽셀 분할 정확도가 아닙니다.
              </InfoBanner>
            </Card>
          </div>
        </>
      ) : (
        <>
          <div className="metric-grid four">
            <Metric
              label="선택 모델"
              value="U-Net"
              detail="검증 Recall 기준"
              icon={ScanLine}
              tone="blue"
            />
            <Metric
              label="Test Dice"
              value="0.478"
              detail="픽셀 단위"
              icon={Activity}
              tone="green"
            />
            <Metric
              label="Test IoU"
              value="0.314"
              detail="픽셀 단위"
              icon={Grid3X3}
              tone="slate"
            />
            <Metric
              label="Test Recall"
              value="96.9%"
              detail="결함 이미지 기준"
              icon={ShieldCheck}
              tone="amber"
            />
          </div>
          <Card
            title="동일 홀드아웃 모델 비교"
            action={<Pill tone="neutral">원본 120장</Pill>}
          >
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>모델</th>
                    <th>Test Dice</th>
                    <th>Test IoU</th>
                    <th>Recall</th>
                    <th>FN</th>
                    <th>FP</th>
                    <th>중앙 처리 시간</th>
                    <th>상태</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <strong>U-Net</strong>
                      <small>BCE + Dice</small>
                    </td>
                    <td>0.478</td>
                    <td>0.314</td>
                    <td>
                      <strong className="good">96.9%</strong>
                    </td>
                    <td>2</td>
                    <td>52</td>
                    <td>36.50 ms</td>
                    <td>
                      <Pill tone="green">
                        <Check size={12} /> 선택
                      </Pill>
                    </td>
                  </tr>
                  <tr>
                    <td>
                      <strong>YOLO Semantic</strong>
                      <small>Ultralytics 8.4</small>
                    </td>
                    <td>0.446</td>
                    <td>0.287</td>
                    <td>93.8%</td>
                    <td>4</td>
                    <td>44</td>
                    <td>44.07 ms</td>
                    <td>
                      <Pill tone="neutral">비교</Pill>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
      <InfoBanner tone="amber">
        <strong>해석 주의</strong> · 분류는 전체 12,568장 분할, 위치 분할은
        570장 부분 실험입니다. 서로 다른 평가 단위의 수치를 직접 비교하지
        마세요.
      </InfoBanner>
    </>
  )
}

export default function App() {
  const [page, setPage] = useState<PageId>("overview")
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [experimentOpen, setExperimentOpen] = useState(false)
  const meta = pageMeta[page]
  const mainRef = useRef<HTMLElement>(null)
  const navigate = (next: PageId) => {
    setPage(next)
    mainRef.current?.scrollTo({ top: 0, behavior: "smooth" })
  }
  return (
    <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Layers3 size={22} />
          </div>
          <div>
            <strong>SteelVision</strong>
            <span>ML INSPECTION</span>
          </div>
        </div>
        <button
          className="experiment-select"
          onClick={() => setExperimentOpen(!experimentOpen)}
        >
          <span>
            <small>현재 실험</small>
            <strong>
              <i className="status-dot" /> Severstal · test1
            </strong>
          </span>
          <ChevronDown size={17} className={experimentOpen ? "rotate" : ""} />
        </button>
        {experimentOpen && (
          <div className="experiment-menu">
            <button>
              <Check size={14} /> Severstal · test1
            </button>
            <span>추가 실험 없음</span>
          </div>
        )}
        <nav>
          <p className="nav-label">WORKSPACE</p>
          {pages.map((item) => {
            const Icon = item.icon
            return (
              <button
                key={item.id}
                className={page === item.id ? "active" : ""}
                onClick={() => navigate(item.id)}
              >
                <Icon size={19} />
                <span>
                  <strong>{item.label}</strong>
                  <small>{item.helper}</small>
                </span>
                {page === item.id && <ChevronRight size={15} />}
              </button>
            )
          })}
        </nav>
        <div className="sidebar-footer">
          <div className="system-card">
            <div>
              <span className="pulse-dot" />
              <strong>실험 데이터 준비됨</strong>
            </div>
            <p>실시간 추론 API는 연결 필요</p>
          </div>
          <button>
            <CircleHelp size={17} /> 사용 가이드
          </button>
        </div>
      </aside>
      <div className="main-column">
        <header className="topbar">
          <button
            className="icon-button sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? <PanelLeftClose size={20} /> : <Menu size={20} />}
          </button>
          <div className="breadcrumb">
            <span>SteelVision ML</span>
            <ChevronRight size={14} />
            <strong>{pages.find((item) => item.id === page)?.label}</strong>
          </div>
          <div className="top-actions">
            <button className="status-button">
              <span className="pulse-dot" /> 실험 데이터 준비
            </button>
            <button className="icon-button">
              <Settings2 size={19} />
            </button>
            <div className="avatar">SV</div>
          </div>
        </header>
        <main ref={mainRef}>
          <div className="page-header">
            <div>
              <span className="eyebrow">{meta.eyebrow}</span>
              <h1>{meta.title}</h1>
              <p>{meta.description}</p>
            </div>
            {page !== "overview" && (
              <div className="updated">
                <Clock3 size={15} />
                <span>
                  저장된 결과
                  <br />
                  <strong>2026. 11. 03 실험</strong>
                </span>
              </div>
            )}
          </div>
          <div className="page-content">
            {page === "overview" && <OverviewPage navigate={navigate} />}
            {page === "eda" && <EdaPage />}
            {page === "preprocess" && <PreprocessPage />}
            {page === "inspection" && <InspectionPage />}
            {page === "localize" && <LocalizePage />}
            {page === "performance" && <PerformancePage />}
          </div>
          <footer>
            SteelVision ML · Severstal baseline workspace{" "}
            <span>실험 결과는 현장 성능을 보증하지 않습니다.</span>
          </footer>
        </main>
      </div>
    </div>
  )
}
