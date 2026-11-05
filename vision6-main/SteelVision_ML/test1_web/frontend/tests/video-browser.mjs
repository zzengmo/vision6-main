import {mkdir,writeFile} from 'node:fs/promises'
import {dirname,resolve,join} from 'node:path'
import {fileURLToPath} from 'node:url'

const app=resolve(dirname(fileURLToPath(import.meta.url)),'..','..')
const output=join(app,'artifacts/verification/video_home')
await mkdir(output,{recursive:true})
const targets=await(await fetch(`http://127.0.0.1:${process.env.STEELVISION_DEBUG_PORT||'9230'}/json`)).json()
const target=targets.find(t=>t.type==='page')
const ws=new WebSocket(target.webSocketDebuggerUrl)
await new Promise((ok,fail)=>{ws.addEventListener('open',ok,{once:true});ws.addEventListener('error',fail,{once:true})})
let id=0;const pending=new Map(),errors=[],mediaResponses=[],postRequests=[]
ws.addEventListener('message',event=>{const value=JSON.parse(event.data);if(value.id){const job=pending.get(value.id);if(job){pending.delete(value.id);value.error?job.reject(Error(JSON.stringify(value.error))):job.resolve(value.result)}}else if(value.method==='Runtime.exceptionThrown'){errors.push(value.params.exceptionDetails)}else if(value.method==='Network.responseReceived'&&value.params.response.url.includes('steel-hero-')&&value.params.response.url.includes('.mp4')){mediaResponses.push({url:value.params.response.url,status:value.params.response.status})}else if(value.method==='Network.requestWillBeSent'&&value.params.request.method==='POST'){postRequests.push(value.params.request.url)}})
function call(method,params={}){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))})}
async function evaluate(expression){const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value}
async function until(expression,seconds=20){const deadline=Date.now()+seconds*1000;while(Date.now()<deadline){if(await evaluate('!!('+expression+')'))return;await new Promise(r=>setTimeout(r,100))}throw Error('Timeout: '+expression)}
async function click(selector){const point=await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw Error('Missing control');const r=e.getBoundingClientRect();return{x:r.left+r.width/2,y:r.top+r.height/2}})()`);await call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});await call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point})}
async function screenshot(name){const result=await call('Page.captureScreenshot',{format:'png'});await writeFile(join(output,name+'.png'),Buffer.from(result.data,'base64'))}
async function navigate(suffix=''){await call('Page.navigate',{url:'http://127.0.0.1:8502/?video-check='+Date.now()+suffix});await until("document.querySelector('.sv-hero-video')")}
const checks=[];const sizes={}
try{
 await call('Page.enable');await call('Runtime.enable');await call('Network.enable');await call('Network.setCacheDisabled',{cacheDisabled:true})
 await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false})
 await navigate();await until("document.querySelector('video').currentTime>.6&&!document.querySelector('video').paused&&document.querySelector('video').readyState>=3")
 sizes.desktop=await evaluate("(()=>{let v=document.querySelector('video');return{width:v.videoWidth,height:v.videoHeight,duration:v.duration,muted:v.muted,currentSrc:v.currentSrc}})()")
 if(!sizes.desktop.muted||!sizes.desktop.currentSrc.endsWith('steel-hero-desktop.mp4'))throw Error('Desktop default source/mute mismatch')
 checks.push('Root URL opens the video homepage with actual muted autoplay')
 await evaluate("document.querySelector('video').currentTime=4")
 await until("document.querySelector('video').currentTime>=4.1");await screenshot('desktop-video')
 await click('.sv-media-button');await until("document.querySelector('video').paused")
 const pausedTime=await evaluate("document.querySelector('video').currentTime")
 await new Promise(r=>setTimeout(r,450));if(Math.abs(await evaluate("document.querySelector('video').currentTime")-pausedTime)>.05)throw Error('Pause did not stop playback')
 await click('.sv-media-button');await until("!document.querySelector('video').paused")
 await click('.sv-media-button:nth-of-type(2)');await until("!document.querySelector('video').muted&&!document.querySelector('video').paused")
 await click('.sv-media-button:nth-of-type(2)');await until("document.querySelector('video').muted")
 checks.push('Trusted clicks pause/resume the video and toggle actual audio mute')
 await click('.sv-header .sv-menu-toggle');await until("document.querySelector('dialog').open&&document.querySelector('video').paused")
 await call('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});await until("!document.querySelector('dialog').open&&!document.querySelector('video').paused")
 checks.push('Opening the menu pauses the video; Escape closes the menu and resumes playback')
 await evaluate("window.scrollTo({top:innerHeight+400,behavior:'instant'})");await until("document.querySelector('video').paused")
 await evaluate("window.scrollTo({top:0,behavior:'instant'})");await until("!document.querySelector('video').paused")
 checks.push('Video pauses outside the hero viewport and resumes on return')
 await evaluate("(()=>{let v=document.querySelector('video');v.currentTime=v.duration-.35})()")
 await until("document.querySelector('video').currentTime<2&&!document.querySelector('video').paused")
 checks.push('Actual playback loops at the end of the clip')
 await click('.sv-floating-workspace');await until("document.querySelector('#quality-model')")
 if(await evaluate("document.querySelectorAll('nav button').length")!==6)throw Error('Existing dashboard menus missing')
 if(await evaluate("!!document.querySelector('video')"))throw Error('Video remained after entering dashboard')
 checks.push('Quality inspection deep link opens the existing six-menu dashboard')
 await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true})
 await navigate();await until("document.querySelector('video').currentSrc.endsWith('steel-hero-mobile.mp4')&&!document.querySelector('video').paused&&document.querySelector('video').currentTime>.5")
 sizes.mobile=await evaluate("(()=>{let v=document.querySelector('video');return{width:v.videoWidth,height:v.videoHeight,duration:v.duration,muted:v.muted,currentSrc:v.currentSrc,playsInline:v.playsInline}})()")
 if(sizes.mobile.width>=sizes.mobile.height||!sizes.mobile.playsInline)throw Error('Mobile portrait/inline playback missing')
 await evaluate("document.querySelector('video').currentTime=4");await until("document.querySelector('video').currentTime>=4.1");await screenshot('mobile-video')
 if(await evaluate('document.documentElement.scrollWidth>innerWidth'))throw Error('Mobile horizontal overflow')
 checks.push('390px mobile uses the separate portrait video and inline playback')
 await call('Emulation.setDeviceMetricsOverride',{width:320,height:720,deviceScaleFactor:1,mobile:true})
 if(await evaluate('document.documentElement.scrollWidth>innerWidth'))throw Error('320px horizontal overflow')
 await screenshot('small-mobile-video');checks.push('320px layout has no horizontal overflow')
 await call('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]})
 await navigate();await until("document.querySelector('video').paused&&!document.querySelector('video').autoplay")
 await screenshot('reduced-motion-poster');await click('.sv-media-button');await until("!document.querySelector('video').paused")
 checks.push('Reduced-motion preference starts with a still poster; deliberate play remains available')
 await call('Emulation.setEmulatedMedia',{features:[]});await call('Network.setBlockedURLs',{urls:['*steel-hero-*.mp4']})
 await navigate();await until("document.querySelector('.sv-video-status.is-fallback')")
 await screenshot('video-error-poster')
 await call('Network.setBlockedURLs',{urls:[]});await click('.sv-media-button');await until("!document.querySelector('video').paused&&document.querySelector('video').currentTime>.3")
 checks.push('Blocked media shows the poster and an honest fallback; retry recovers actual playback')
 if(errors.length)throw Error('Browser runtime exceptions: '+JSON.stringify(errors))
 if(postRequests.length)throw Error('Homepage unexpectedly triggered model inference')
 if(!mediaResponses.some(r=>r.status===206))throw Error('No actual byte-range media response was observed')
 const record={checks,sizes,media_responses:mediaResponses,no_automatic_inference:true,runtime_errors:errors,browser:'Installed Chrome via CDP'}
 await writeFile(join(output,'checks.json'),JSON.stringify(record,null,2));console.log(JSON.stringify(record,null,2))
}finally{ws.close()}
