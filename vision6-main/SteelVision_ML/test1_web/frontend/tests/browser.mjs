import {mkdir,writeFile,readdir,stat,unlink} from 'node:fs/promises'
import {dirname,join,resolve} from 'node:path'
import {fileURLToPath} from 'node:url'

const app=resolve(dirname(fileURLToPath(import.meta.url)),'..','..')
const baseURL=process.env.STEELVISION_TEST_URL||'http://127.0.0.1:8502'
const debugPort=process.env.STEELVISION_DEBUG_PORT||'9230'
const output=join(app,'artifacts','verification','workspace_design')
const downloads=join(output,'downloads')
await mkdir(downloads,{recursive:true})
for(const name of ['preprocessed.png','inspection.csv','inspection_original.png','unet_overlay.png','unet_mask.png','unet_candidates.png','unet_boxes.csv'])await unlink(join(downloads,name)).catch(error=>{if(error.code!=='ENOENT')throw error})
const targets=await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()
const target=targets.find(t=>t.type==='page'&&t.url.startsWith(baseURL))||targets.find(t=>t.type==='page')
if(!target)throw Error(`Start the local QA Chrome on port ${debugPort} first.`)
const socket=new WebSocket(target.webSocketDebuggerUrl)
await new Promise((ok,fail)=>{socket.addEventListener('open',ok,{once:true});socket.addEventListener('error',fail,{once:true})})
let sequence=0;const pending=new Map(),errors=[],requests=[]
socket.addEventListener('message',event=>{const value=JSON.parse(event.data);if(value.id){const p=pending.get(value.id);if(p){pending.delete(value.id);value.error?p.reject(Error(JSON.stringify(value.error))):p.resolve(value.result)}}else if(value.method==='Runtime.exceptionThrown'){errors.push(value.params.exceptionDetails)}else if(value.method==='Network.requestWillBeSent'&&value.params.request.method==='POST'){requests.push(value.params.request.url)}})
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const value=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(value.exceptionDetails)throw Error(JSON.stringify(value.exceptionDetails));return value.result.value}
async function until(expression,seconds=45){const deadline=Date.now()+seconds*1000;while(Date.now()<deadline){if(await evaluate('!!('+expression+')'))return;await new Promise(r=>setTimeout(r,200))}throw Error('Timed out: '+expression)}
async function clickText(selector,text){await evaluate(`(()=>{const b=[...document.querySelectorAll(${JSON.stringify(selector)})].find(x=>x.textContent.includes(${JSON.stringify(text)}));if(!b)throw Error('Missing button');b.click()})()`)}
async function page(label){await clickText('nav button',label);await new Promise(r=>setTimeout(r,200))}
async function file(label){const tree=await call('DOM.getDocument');const node=await call('DOM.querySelector',{nodeId:tree.root.nodeId,selector:'input[type=file]'});await call('DOM.setFileInputFiles',{nodeId:node.nodeId,files:[join(app,'artifacts','verification',label,'input.png')]});await until(`document.querySelector('.upload-copy strong')?.textContent==='input.png'`)}
async function screenshot(name){const value=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(join(output,name+'.png'),Buffer.from(value.data,'base64'))}
async function noOverflow(label){if(await evaluate('document.documentElement.scrollWidth>innerWidth'))throw Error('Horizontal overflow: '+label)}
async function download(name){await evaluate(`document.querySelector('a[download=${JSON.stringify(name)}]').click()`);const deadline=Date.now()+10000;while(Date.now()<deadline){if((await readdir(downloads)).includes(name))return;await new Promise(r=>setTimeout(r,100))}throw Error('Download missing: '+name)}
const checks=[]
try{
 await call('Runtime.enable');await call('Page.enable');await call('Network.enable')
 await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads,eventsEnabled:true})
 await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false})
 await call('Page.navigate',{url:baseURL+'/?workspace-check='+Date.now()+'#/dashboard/overview'})
 await until(`document.querySelectorAll('nav button').length===6 && document.querySelector('.overview-hero')`)
 await screenshot('01_overview');await noOverflow('desktop overview');checks.push('All six navigation menus present; corporate header, photo masthead and horizontal tabs rendered')
 await page('데이터 EDA');await until(`document.querySelectorAll('tbody tr').length>=7`);await screenshot('02_eda')
 await page('이미지 전처리 체험');await file('normal');await clickText('.run-button','전처리 실행');await until(`document.querySelector('.histogram')`);await download('preprocessed.png');await screenshot('03_preprocess');checks.push('Uploaded image processed; histogram and real PNG downloaded')
 await page('자동 품질 검수');await file('normal');await clickText('.run-button','검수 실행');await until(`document.querySelector('a[download="inspection.csv"]')`)
 if(!(await evaluate(`document.querySelector('.result-card').innerText.includes('정상')`)))throw Error('Normal TTA decision missing')
 const count=requests.filter(x=>x.includes('/api/inspect')).length
 await download('inspection.csv');await download('inspection_original.png')
 if(!(await evaluate(`!!document.querySelector('a[download="inspection.csv"]')`)))throw Error('Result disappeared after download')
 if(requests.filter(x=>x.includes('/api/inspect')).length!==count)throw Error('Download triggered inference')
 checks.push('Actual ResNet18 TTA normal classification; CSV/PNG download retains result without duplicate inference')
 await file('defect');await until(`!document.querySelector('a[download="inspection.csv"]')`)
 await clickText('.run-button','검수 실행');await until(`document.querySelector('a[download="inspection.csv"]')`)
 if(!(await evaluate(`document.querySelector('.result-card').innerText.includes('불량 의심')`)))throw Error('Defect TTA decision missing')
 await screenshot('04_inspection');await evaluate(`document.querySelector('.result-card').scrollIntoView({block:'start'})`);await screenshot('04_inspection_result');checks.push('Changing image clears old result; actual defect TTA classification succeeds')
 await evaluate(`document.querySelector('.sv-header .sv-menu-toggle').click()`);await until(`document.querySelector('dialog').open`)
 await call('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});await until(`!document.querySelector('dialog').open`)
 if(!(await evaluate(`!!document.querySelector('a[download="inspection.csv"]')`)))throw Error('Menu removed the inspection result')
 checks.push('Shared fullscreen menu opens and Escape closes it without losing the inspection result')
 await evaluate(`(()=>{let x=document.querySelector('#quality-model');x.value='rf_current';x.dispatchEvent(new Event('change',{bubbles:true}))})()`)
 await until(`!document.querySelector('a[download="inspection.csv"]')`);checks.push('Changing model invalidates the previous result')
 await page('정밀 결함 위치');await file('defect');await clickText('.run-button','결함 위치 찾기');await until(`document.querySelector('a[download="unet_mask.png"]')`)
 for(const name of ['unet_overlay.png','unet_mask.png','unet_candidates.png','unet_boxes.csv'])await download(name)
 await screenshot('05_localize');checks.push('Actual U-Net localization succeeds; overlay/mask/candidate PNG and coordinates CSV downloaded')
 await page('모델 성능 비교');await screenshot('06_performance');await clickText('.report-tabs button','딥러닝 노트북 비교');await until(`document.querySelector('tbody')?.textContent.includes('99.70%')`)
 checks.push('Stored deep comparison is separated from current PC evaluation')
 await evaluate(`(()=>{const x=document.querySelector('#experiment');x.value='reference';x.dispatchEvent(new Event('change',{bubbles:true}))})()`)
 for(const label of ['프로젝트 소개','데이터 EDA','이미지 전처리 체험','자동 품질 검수','정밀 결함 위치','모델 성능 비교']){await page(label);await noOverflow('reference '+label)}
 checks.push('All six reference-mode pages render; actual inspection stays disabled in reference mode')
 await evaluate(`(()=>{const x=document.querySelector('#experiment');x.value='current';x.dispatchEvent(new Event('change',{bubbles:true}))})()`)
 await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true})
 for(const [label,name] of [['프로젝트 소개','07_mobile_overview'],['데이터 EDA','08_mobile_eda'],['이미지 전처리 체험','09_mobile_preprocess'],['자동 품질 검수','10_mobile_inspection'],['정밀 결함 위치','11_mobile_localize'],['모델 성능 비교','12_mobile_performance']]){await page(label);await noOverflow('390px '+label);await screenshot(name)}
 await evaluate(`document.querySelector('.sv-header .sv-menu-toggle').click()`);await until(`document.querySelector('dialog').open`);await screenshot('13_mobile_menu')
 await evaluate(`document.querySelector('.sv-menu-groups a[href="#/dashboard/inspection"]').click()`);await until(`document.querySelector('#quality-model')&&!document.querySelector('dialog').open`)
 checks.push('390px: all six pages fit the viewport; shared fullscreen menu navigates to quality inspection')
 await call('Emulation.setDeviceMetricsOverride',{width:320,height:720,deviceScaleFactor:1,mobile:true})
 for(const label of ['프로젝트 소개','데이터 EDA','이미지 전처리 체험','자동 품질 검수','정밀 결함 위치','모델 성능 비교']){await page(label);await noOverflow('320px '+label)}
 await page('자동 품질 검수');await screenshot('14_small_mobile_inspection')
 const overlaps=await evaluate(`(()=>{const logo=document.querySelector('.sv-header .sv-logo').getBoundingClientRect(),menu=document.querySelector('.sv-header .sv-menu-toggle').getBoundingClientRect();return logo.right>menu.left})()`)
 if(overlaps)throw Error('320px header logo overlaps the menu')
 checks.push('320px: all six pages avoid horizontal overflow; header logo and menu do not overlap')
 if(errors.length)throw Error('Browser runtime errors: '+JSON.stringify(errors))
 const files=await readdir(downloads);const sizes=await Promise.all(files.map(async f=>({file:f,bytes:(await stat(join(downloads,f))).size})))
 await writeFile(join(output,'checks.json'),JSON.stringify({checks,errors,post_requests:requests,downloads:sizes,browser:'Installed Chrome via CDP',server:baseURL},null,2))
 console.log(JSON.stringify({checks,downloaded:files,errors},null,2))
}finally{socket.close()}
