import {useEffect,useRef,type ReactNode} from 'react'
import {ArrowUpRight,ChevronRight,RefreshCw} from 'lucide-react'
import {SiteHeader,SiteFooter} from './SiteChrome'
import {workspacePages,workspaceMeta,type WorkspacePage} from './workspace-navigation'
import './workspace.css'

interface Props{
  page:WorkspacePage
  mode:string
  onMode:(mode:string)=>void
  onNavigate:(page:WorkspacePage)=>void
  onReload:()=>void
  connected:boolean
  error:boolean
  readyModels:number|null
  children:ReactNode
}

export default function WorkspaceShell({page,mode,onMode,onNavigate,onReload,connected,error,readyModels,children}:Props){
  const meta=workspaceMeta[page],number=String(workspacePages.findIndex(p=>p.id===page)+1).padStart(2,'0')
  const nav=useRef<HTMLElement>(null)
  useEffect(()=>{
    const container=nav.current,active=container?.querySelector('[aria-current="page"]')
    if(window.innerWidth>=900||!container||!active)return
    const bounds=container.getBoundingClientRect(),item=active.getBoundingClientRect()
    container.scrollTo({left:container.scrollLeft+item.left-bounds.left-(container.clientWidth-item.width)/2,behavior:'auto'})
  },[page])
  return <div className="sv-workspace">
    <a className="ws-skip" href="#workspace-content" onClick={event=>{event.preventDefault();document.getElementById('workspace-content')?.focus()}}>본문 바로가기</a>
    <SiteHeader solid activePage={page}/>
    <section className="ws-masthead" aria-labelledby="workspace-title"><div className="ws-masthead-inner">
      <div className="ws-breadcrumb"><a href="#/intro">HOME</a><ChevronRight size={12}/><a href="#/dashboard/overview">WORKSPACE</a><ChevronRight size={12}/><span>{workspacePages.find(p=>p.id===page)?.label}</span></div>
      <div className="ws-page-heading"><div><span className="ws-eyebrow">{meta.eyebrow}</span><h1 id="workspace-title">{meta.title}</h1><p>{meta.description}</p></div><span className="ws-section-number" aria-hidden="true">{number}</span></div>
    </div></section>
    <nav ref={nav} className="ws-tabs" aria-label="검수 기능 메뉴"><div className="ws-tabs-inner">{workspacePages.map(({id,label,icon:Icon},index)=><button key={id} className={page===id?'active':''} aria-current={page===id?'page':undefined} onClick={()=>onNavigate(id)}><span className="ws-tab-number">{String(index+1).padStart(2,'0')}</span><Icon size={17}/><strong>{label}</strong></button>)}</div></nav>
    <main id="workspace-content" className="ws-main" tabIndex={-1}>
      <div className="ws-source-bar"><div className="ws-source-select"><label htmlFor="experiment"><span>EXPERIMENT</span>실험 결과 선택</label><select id="experiment" value={mode} onChange={event=>onMode(event.target.value)}><option value="current">현재 PC 재현 결과</option><option value="reference">노트북 저장 결과 (참고)</option></select></div><div className="ws-connection"><span className={`ws-status-dot ${error?'is-error':''}`}/><span>{error?'연결 확인 필요':connected?`분류 모델 ${readyModels}개 사용 가능`:'모델 준비 상태 확인 중'}</span><button aria-label="모델 상태 새로 확인" title="모델 상태 새로 확인" onClick={onReload}><RefreshCw size={15}/></button></div></div>
      <div className="page-content">{children}</div>
      <div className="ws-workspace-end"><span>STEELVISION ML</span><p>작은 차이를 발견하고, 실제 결과로 판단합니다.</p><a href="#/intro">영상 메인으로<ArrowUpRight size={17}/></a></div>
    </main>
    <SiteFooter/>
  </div>
}
