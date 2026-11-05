import {useEffect,useState} from 'react'
import App from './App'
import Landing from './Landing'

export default function Entry(){
  const [intro,setIntro]=useState(()=>!window.location.hash.startsWith('#/dashboard'))
  useEffect(()=>{const route=()=>{setIntro(!window.location.hash.startsWith('#/dashboard'));window.scrollTo(0,0)};window.addEventListener('hashchange',route);return()=>window.removeEventListener('hashchange',route)},[])
  return intro?<Landing/>:<App/>
}
