import {useEffect, useRef, useState} from 'react'
import {ArrowDown, ArrowUpRight, Database, Pause, Play, ScanLine, ShieldCheck, Volume2, VolumeX} from 'lucide-react'
import {SiteHeader,SiteFooter} from './SiteChrome'
import {workspaceHref as href} from './workspace-navigation'
import './landing.css'

const timeLabel=(seconds:number)=>`${Math.floor(seconds/60).toString().padStart(2,'0')}:${Math.floor(seconds%60).toString().padStart(2,'0')}`
export default function Landing(){
  const [paused,setPaused]=useState(()=>matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [mobile,setMobile]=useState(()=>matchMedia('(max-width:700px)').matches)
  const [muted,setMuted]=useState(true),[playing,setPlaying]=useState(false),[hasFrame,setHasFrame]=useState(false),[failed,setFailed]=useState(false)
  const [elapsed,setElapsed]=useState(0),[duration,setDuration]=useState(0)
  const [heroVisible,setHeroVisible]=useState(true),[pageVisible,setPageVisible]=useState(()=>!document.hidden)
  const [menu,setMenu]=useState(false),[scrolled,setScrolled]=useState(false),[data,setData]=useState<any>(null)
  const video=useRef<HTMLVideoElement>(null),hero=useRef<HTMLElement>(null)
  const variant=mobile?'mobile':'desktop',videoSrc=`/landing/steel-hero-${variant}.mp4`,poster=`/landing/steel-hero-${variant}.jpg`
  const shouldPlay=!paused&&!menu&&heroVisible&&pageVisible&&!failed
  useEffect(()=>{const onScroll=()=>setScrolled(window.scrollY>40);window.addEventListener('scroll',onScroll,{passive:true});onScroll();return()=>window.removeEventListener('scroll',onScroll)},[])
  useEffect(()=>{const query=matchMedia('(max-width:700px)'),motion=matchMedia('(prefers-reduced-motion: reduce)');const resize=()=>setMobile(query.matches),reduce=()=>{if(motion.matches)setPaused(true)},visibility=()=>setPageVisible(!document.hidden);query.addEventListener('change',resize);motion.addEventListener('change',reduce);document.addEventListener('visibilitychange',visibility);return()=>{query.removeEventListener('change',resize);motion.removeEventListener('change',reduce);document.removeEventListener('visibilitychange',visibility)}},[])
  useEffect(()=>{if(!hero.current)return;const observer=new IntersectionObserver(([entry])=>setHeroVisible(entry.isIntersecting),{threshold:.15});observer.observe(hero.current);return()=>observer.disconnect()},[])
  useEffect(()=>{setFailed(false);setHasFrame(false);setElapsed(0);setDuration(0)},[videoSrc])
  useEffect(()=>{const element=video.current;if(!element)return;let active=true;element.volume=.45;if(shouldPlay){element.play().catch((error:DOMException)=>{if(active&&error.name!=='AbortError')setPaused(true)})}else element.pause();return()=>{active=false}},[shouldPlay,videoSrc])
  useEffect(()=>{const controller=new AbortController();fetch('/api/metadata',{signal:controller.signal}).then(r=>r.ok?r.json():null).then(setData).catch(()=>{});return()=>controller.abort()},[])

  function scrollToQuality(){document.getElementById('sv-quality')?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'})}
  function togglePlayback(){
    const element=video.current
    if(!element)return
    if(paused||failed){
      if(failed){setFailed(false);element.load()}
      setPaused(false)
      // Keep the native play call within the user gesture for mobile browsers.
      element.play().catch((error:DOMException)=>{if(error.name!=='AbortError')setPaused(true)})
    }else{setPaused(true);element.pause()}
  }
  function toggleMute(){
    const next=!muted
    if(video.current)video.current.muted=next
    setMuted(next)
  }
  return <div className="sv-home">
    <a className="sv-skip" href="#sv-quality" onClick={event=>{event.preventDefault();scrollToQuality()}}>본문 바로가기</a>
    <SiteHeader solid={scrolled} onMenuChange={setMenu}/>
    <section ref={hero} className="sv-hero sv-video-hero" aria-label="철강 품질 프로젝트 소개 영상">
      <div className="sv-video-poster" style={{backgroundImage:`url(${poster})`}} aria-hidden="true"/>
      <video key={variant} ref={video} className={`sv-hero-video ${hasFrame&&!failed?'has-frame':''}`} src={videoSrc} poster={poster} autoPlay={shouldPlay} muted={muted} loop playsInline preload="metadata" disablePictureInPicture aria-hidden="true" tabIndex={-1}
        onPlaying={()=>{setPlaying(true);setHasFrame(true)}} onPause={()=>setPlaying(false)} onWaiting={()=>setPlaying(false)}
        onLoadedMetadata={event=>setDuration(Number.isFinite(event.currentTarget.duration)?event.currentTarget.duration:0)}
        onTimeUpdate={event=>setElapsed(event.currentTarget.currentTime)} onError={()=>{setFailed(true);setPlaying(false)}}/>
      <div className="sv-hero-shade"/>
      <div className="sv-hero-copy"><span className="sv-eyebrow">STEEL MEETS INTELLIGENCE</span><h1>철강의 품질,<br/><span>기술로 완성하다.</span></h1><p>표면의 작은 차이에서 시작하는 품질의 변화.<br/>데이터 탐색에서 실제 결함 검수까지, SteelVision ML.</p><a className="sv-outline-link" href={href('overview')}>SteelVision 알아보기<ArrowUpRight size={19}/></a></div>
      <a className="sv-floating-workspace" href={href('inspection')}><ScanLine size={31}/><span>품질 검수<br/>워크스페이스</span><ArrowUpRight size={18}/></a>
      <div className="sv-hero-bottom"><div className="sv-video-controls"><div className="sv-video-timeline"><span>STEEL IN MOTION</span><div className="sv-video-progress" aria-hidden="true"><i style={{width:`${duration?Math.min(100,elapsed/duration*100):0}%`}}/></div><small>{timeLabel(elapsed)} <i>/ {timeLabel(duration)}</i></small></div><button className="sv-media-button" aria-label={failed?'배경 영상 다시 불러오기':paused?'배경 영상 재생':'배경 영상 일시정지'} aria-pressed={paused} onClick={togglePlayback}>{paused||failed?<Play size={17}/>:<Pause size={17}/>}</button><button className="sv-media-button" aria-label={muted?'배경 영상 음소거 해제':'배경 영상 음소거'} aria-pressed={muted} onClick={toggleMute}>{muted?<VolumeX size={17}/>:<Volume2 size={17}/>}</button></div><button className="sv-scroll" onClick={scrollToQuality}><span>SCROLL TO EXPLORE</span><ArrowDown size={18}/></button><a className="sv-hero-cta" href={href('inspection')}>품질 검수 시작하기<ArrowUpRight size={19}/></a></div>
      <span className={`sv-video-status ${failed?'is-fallback':''}`} role="status">{failed?'영상 대신 이미지를 표시합니다. 다시 재생할 수 있습니다.':paused?'배경 영상 일시정지':playing?'':shouldPlay?'영상을 준비하고 있습니다.':''}</span>
    </section>
    <section className="sv-quality" id="sv-quality">
      <div className="sv-section-heading"><span className="sv-eyebrow">OUR TECHNOLOGY</span><h2>데이터에서 판단으로,<br/><em>철강 품질의 새로운 시선.</em></h2><p>데이터를 살펴보고, 품질을 판정하고, 결함의 위치를 확인합니다.<br className="sv-desktop-only"/>각 모델이 실제로 지원하는 결과를 하나의 작업 공간에서 제공합니다.</p></div>
      <div className="sv-technology-grid">{[
        {number:'01',icon:Database,title:'데이터를 이해하다',english:'DATA EXPLORER',description:'실제 데이터 분포와 원본 사진, CSV 정답 주석을 확인합니다.',page:'eda'},
        {number:'02',icon:ShieldCheck,title:'품질을 판단하다',english:'QUALITY INSPECTION',description:'RF·ResNet18으로 이미지 전체의 정상·불량을 판정합니다.',page:'inspection'},
        {number:'03',icon:ScanLine,title:'위치를 발견하다',english:'DEFECT LOCALIZATION',description:'U-Net이 예측한 마스크와 원본 좌표를 확인합니다.',page:'localize'},
      ].map(item=><a className="sv-technology" href={href(item.page)} key={item.number}><span className="sv-technology-number">{item.number}</span><item.icon size={37} strokeWidth={1.2}/><span className="sv-technology-english">{item.english}</span><h3>{item.title}</h3><p>{item.description}</p><span className="sv-technology-arrow"><ArrowUpRight size={24}/></span></a>)}</div>
      <div className="sv-data-strip"><div><span>저장 데이터 기준 원본 이미지</span><strong>{data?.rf?.sample_count?.toLocaleString()??'—'}<small>장</small></strong></div><div><span>RF 학습 분할</span><strong>{data?.rf?.splits?.train?.toLocaleString()??'—'}<small>장</small></strong></div><div><span>실제 사용 가능한 분류 모델</span><strong>{data?.models?data.models.filter((m:any)=>m.ready).length:'—'}<small>개</small></strong></div><a href={href('performance')}>평가 결과 살펴보기<ArrowUpRight size={20}/></a></div>
    </section>
    <section className="sv-future"><div className="sv-future-bg"/><div className="sv-future-content"><span className="sv-eyebrow">BETTER QUALITY, BETTER TOMORROW</span><h2>더 세밀하게 보고,<br/>더 신뢰할 수 있게.</h2><p>품질 판정과 결함 위치를 구분하고,<br/>실제 결과와 저장된 평가를 함께 확인합니다.</p><a className="sv-outline-link" href={href('performance')}>모델 리포트 보기<ArrowUpRight size={20}/></a></div></section>
    <SiteFooter/>
  </div>
}
