import {useEffect,useRef,useState} from 'react'
import {ArrowRight,ArrowUp,ArrowUpRight,Menu,X} from 'lucide-react'
import {workspaceGroups,workspaceHref,type WorkspacePage} from './workspace-navigation'

export function SiteHeader({solid=false,activePage,onMenuChange}:{solid?:boolean;activePage?:WorkspacePage;onMenuChange?:(open:boolean)=>void}){
  const [open,setOpen]=useState(false)
  const dialog=useRef<HTMLDialogElement>(null)
  useEffect(()=>{onMenuChange?.(open)},[open,onMenuChange])
  useEffect(()=>{
    const element=dialog.current
    if(!element)return
    if(!open){element.close();return}
    element.showModal()
    const previous=document.body.style.overflow
    document.body.style.overflow='hidden'
    return()=>{document.body.style.overflow=previous;element.close()}
  },[open])
  return <>
    <header className={`sv-header ${solid?'sv-header-solid':''}`}>
      <a className="sv-logo" href="#/intro" aria-label="SteelVision 소개 홈">steelvision<span>ML</span></a>
      <button className="sv-menu-toggle" aria-label="전체 메뉴 열기" aria-expanded={open} aria-controls="sv-full-menu" onClick={()=>setOpen(true)}><Menu size={25}/></button>
      <div className="sv-header-links"><span className="sv-language">KOR</span><a href={workspaceHref('inspection')}>검수 시작<ArrowUpRight size={15}/></a></div>
    </header>
    <dialog ref={dialog} id="sv-full-menu" className="sv-menu" onCancel={event=>{event.preventDefault();setOpen(false)}}>
      <div className="sv-menu-top"><a className="sv-logo" href="#/intro" onClick={()=>setOpen(false)}>steelvision<span>ML</span></a><button className="sv-menu-toggle" aria-label="전체 메뉴 닫기" onClick={()=>setOpen(false)}><X size={25}/></button><span>STEEL QUALITY WORKSPACE</span></div>
      <div className="sv-menu-groups">{workspaceGroups.map(group=><section key={group.title}><span>{group.number}</span><h2>{group.title}</h2>{group.links.map(([label,page])=><a href={workspaceHref(page)} key={label} aria-current={activePage===page?'page':undefined} onClick={()=>setOpen(false)}>{label}<ArrowUpRight size={17}/></a>)}</section>)}</div>
      <div className="sv-menu-bottom"><p>표면의 작은 차이에서 시작하는 품질의 변화.</p><a href={workspaceHref('overview')} onClick={()=>setOpen(false)}>전체 워크스페이스<ArrowRight size={18}/></a></div>
    </dialog>
  </>
}

export function SiteFooter(){return <footer className="sv-footer">
  <div className="sv-footer-main"><div><a className="sv-logo" href="#/intro">steelvision<span>ML</span></a><p>철강 표면 결함 분석 프로젝트<br/>데이터 탐색에서 실제 이미지 검수까지.</p></div><div className="sv-footer-links">{workspaceGroups.map(group=><div key={group.title}><strong>{group.title}</strong>{group.links.slice(0,2).map(([label,page])=><a href={workspaceHref(page)} key={label}>{label}</a>)}</div>)}</div></div>
  <div className="sv-footer-bottom"><span>STEELVISION ML · QUALITY INSPECTION PROJECT</span><a href="https://www.posco.co.kr/homepage/docs/kor7/jsp/s91a0000001i.jsp" target="_blank" rel="noreferrer">영상·사진·디자인 참고: POSCO<ArrowUpRight size={12}/></a><button onClick={()=>window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'})}>TOP<ArrowUp size={14}/></button></div>
</footer>}
