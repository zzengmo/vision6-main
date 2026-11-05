import {writeFile} from 'node:fs/promises'
import {dirname,resolve,join} from 'node:path'
import {fileURLToPath} from 'node:url'
const app=resolve(dirname(fileURLToPath(import.meta.url)),'..','..')
const output=join(app,'artifacts/verification/workspace_design')
const targets=await(await fetch(`http://127.0.0.1:${process.env.STEELVISION_DEBUG_PORT||'9230'}/json`)).json()
const ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl)
await new Promise(r=>ws.addEventListener('open',r,{once:true}))
let id=0;const tasks=new Map()
ws.addEventListener('message',e=>{const r=JSON.parse(e.data);if(r.id){tasks.get(r.id)(r.result);tasks.delete(r.id)}})
const call=(method,params={})=>new Promise(r=>{tasks.set(++id,r);ws.send(JSON.stringify({id,method,params}))})
async function evaluate(expression){const r=await call('Runtime.evaluate',{returnByValue:true,awaitPromise:true,expression});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
async function until(expression){for(let i=0;i<100;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100))}throw Error(expression)}
const checks=[]
try{
  await call('Page.enable');await call('Runtime.enable')
  for(const width of [1440,900,758,700,390,320]){
    await call('Emulation.setDeviceMetricsOverride',{width,height:width<700?844:1000,deviceScaleFactor:1,mobile:width<700})
    await call('Page.navigate',{url:'http://127.0.0.1:8502/?layout-check='+Date.now()+'#/dashboard/inspection'})
    await until(`!!document.querySelector('#quality-model')`)
    if(await evaluate('scrollY>1'))throw Error('Direct link scrolled away from the banner: '+width)
    for(const page of ['overview','eda','preprocess','inspection','localize','performance']){
      await evaluate(`document.querySelector('nav button:nth-child(${['overview','eda','preprocess','inspection','localize','performance'].indexOf(page)+1})').click()`)
      await until(`location.hash==='#/dashboard/${page}'&&document.querySelector('nav [aria-current=page]').textContent.includes(${JSON.stringify({overview:'프로젝트 소개',eda:'데이터 EDA',preprocess:'이미지 전처리 체험',inspection:'자동 품질 검수',localize:'정밀 결함 위치',performance:'모델 성능 비교'}[page])})`)
      await new Promise(r=>setTimeout(r,120))
      if(await evaluate('document.documentElement.scrollWidth>innerWidth'))throw Error('Overflow: '+width+' '+page)
      if(await evaluate('scrollY>1'))throw Error('Horizontal tabs scrolled the page vertically: '+width+' '+page)
    }
    checks.push({width,pages:6,direct_link_keeps_banner:true,horizontal_overflow:false})
  }
  await evaluate(`document.querySelector('nav button:nth-child(4)').click()`)
  await evaluate(`history.back()`);await until(`location.hash==='#/dashboard/performance'`)
  await evaluate(`history.forward()`);await until(`location.hash==='#/dashboard/inspection'&&!!document.querySelector('#quality-model')`)
  await evaluate(`(()=>{const x=document.querySelector('#experiment');x.value='reference';x.dispatchEvent(new Event('change',{bubbles:true}))})()`)
  await until(`document.querySelector('.run-button').disabled`)
  const shot=await call('Page.captureScreenshot',{format:'png'})
  await writeFile(join(output,'15_direct_mobile.png'),Buffer.from(shot.data,'base64'))
  const result={checks,history_back_forward:true,reference_inference_disabled:true}
  await writeFile(join(output,'layout-checks.json'),JSON.stringify(result,null,2))
  console.log(JSON.stringify(result,null,2))
}finally{ws.close()}
