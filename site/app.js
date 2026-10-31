const $=id=>document.getElementById(id);
const DT=0.02, clamp=(v,a,b)=>Math.max(a,Math.min(b,v)), norm=a=>Math.hypot(...a);
let pyodide=null, engineReset=null, engineStep=null, engineDiagnostics=null, profiles=[], profile=null, snapshot=null;
let diagnosticHistory=[];
let scene,camera,renderer,droneMesh,propellers=[], armed=false,paused=false, motors=[0,0,0,0], throttle=0, directMotorMode=false;
let lastFrame=0, accumulator=0, simRunning=false, stepBusy=false, stepsPerFrame=5;
const keys={};
function setStatus(text,ok=false){$("statusText").textContent=text;$("statusDot").style.background=ok?"#56d6a0":"#f0b66a"}
function log(text){$("log").textContent=text}
function diagLog(level,message,code=""){const item={time:new Date().toLocaleTimeString(),level,message,code};diagnosticHistory.push(item);if(diagnosticHistory.length>300)diagnosticHistory.shift();const host=$("diagnosticLog");if(host){host.textContent=diagnosticHistory.map(x=>`[${x.time}] [${x.level.toUpperCase()}] ${x.code?x.code+" · ":""}${x.message}`).join("\n");host.scrollTop=host.scrollHeight}}
function safeJson(value){try{return JSON.stringify(value,null,2)}catch(e){return String(value)}}
function collectFlightDebugBundle(){
 let engineChecks=[];try{if(engineDiagnostics)engineChecks=JSON.parse(engineDiagnostics())}catch(e){engineChecks={error:e.stack||e.message}}
 const state=snapshot?.state||[];
 const info=snapshot?.info||{};
 const params={
  mass_kg:snapshot?.mass_kg??info.mass_kg??info.scenario?.mass_kg??null,
  center_of_mass_m:info.center_of_mass_m??null,
  inertia_kg_m2:info.inertia_kg_m2??null,
  rotor_positions_m:profile?.rotors?.map(r=>r.position_m)??null,
  rotor_spin_signs:info.rotor_reaction_torque_sign??profile?.rotors?.map(r=>r.spin_sign)??null,
  motor_effectiveness:info.motor_effectiveness??info.scenario?.motor_effectiveness??null,
  scenario:info.scenario??snapshot?.info?.scenario??null,
  thrust_model_status:info.thrust_model_status??null
 };
 return {
  report_type:"flight_lab_debug_bundle",schema_version:2,captured_at:new Date().toISOString(),
  page:{url:location.href,user_agent:navigator.userAgent,language:navigator.language,platform:navigator.platform,hardware_concurrency:navigator.hardwareConcurrency??null,device_memory_gb:navigator.deviceMemory??null,screen:{width:screen.width,height:screen.height,pixel_ratio:devicePixelRatio},secure_context:window.isSecureContext},
  configuration:{vehicle_id:profile?.vehicle_id??null,vehicle_name:profile?.display_name??null,seed:Number($("seedInput")?.value),curriculum_stage:Number($("curriculumSelect")?.value),randomization_enabled:$("randomizeToggle")?.checked??null,armed,paused,direct_motor_mode:directMotorMode,throttle,manual_motor_commands:motors,dt_s:DT},
  simulation:{step:snapshot?.step??null,time_s:snapshot?.time_s??null,terminated:snapshot?.terminated??null,truncated:snapshot?.truncated??null,reward:snapshot?.reward??null,state:{position_xyz_m:state.slice(0,3),velocity_xyz_m_s:state.slice(3,6),attitude_rpy_rad:state.slice(6,9),attitude_rpy_deg:state.slice(6,9).map(v=>v*180/Math.PI),angular_velocity_body_rad_s:state.slice(9,12)},snapshot:snapshot??null},
  physical_parameters:params,
  diagnostics:{engine_health_checks:engineChecks,ui_event_history:diagnosticHistory},
  visible_logs:{runtime:$("log")?.textContent||"",engine_meta:$("engineMeta")?.textContent||"",diagnostic_panel:$("diagnosticLog")?.textContent||""},
  interpretation_notes:["This bundle contains simulator model values, not hardware measurements.","For thrust/torque root-cause analysis, retain per-rotor thrust, rotor speed, motor effectiveness, body torque, angular acceleration, CG and inertia fields from simulation.info."]
 };
}
async function copyAllLogs(){
 const bundle=collectFlightDebugBundle();
 const content="=== FLIGHT LAB DEBUG BUNDLE (JSON) ===\n"+safeJson(bundle)+"\n\n=== HUMAN-READABLE DIAGNOSTICS ===\n"+diagnosticHistory.map(x=>"["+(x.time||"")+"] ["+(x.level||"INFO").toUpperCase()+"] "+(x.code?x.code+" · ":"")+(x.message||"")).join("\n");
 try{
  if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(content);
  else{
   const area=document.createElement("textarea");area.value=content;area.setAttribute("readonly","");area.style.position="fixed";area.style.opacity="0";document.body.appendChild(area);area.select();
   const copied=document.execCommand("copy");area.remove();if(!copied)throw Error("مرورگر اجازهٔ کپی نداد");
  }
  const button=$("copyDiagnostics");if(button){button.textContent="گزارش کامل کپی شد";setTimeout(()=>button.textContent="کپی همهٔ لاگ‌ها",2200)}
 }catch(e){diagLog("error","کپی لاگ ناموفق بود: "+(e.message||String(e)),"diagnostics.copy")}
}
function renderEngineDiagnostics(){if(!engineDiagnostics)return;try{const rows=JSON.parse(engineDiagnostics());for(const x of rows)diagLog(x.level||"info",x.message||JSON.stringify(x),x.code||"")}catch(e){diagLog("error",e.stack||e.message,"diagnostics.read")}}
function rotationMatrix(a){const [r,p,y]=a,[cr,sr,cp,sp,cy,sy]=[Math.cos(r),Math.sin(r),Math.cos(p),Math.sin(p),Math.cos(y),Math.sin(y)];return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]}
function threeRotation(a){const R=rotationMatrix(a);return [[R[0][0],-R[0][2],R[0][1]],[ -R[2][0],R[2][2],-R[2][1]],[R[1][0],-R[1][2],R[1][1]]]}
let droneModelRequest=0;
function buildDrone(){
 if(!scene||!profile)return;
 if(droneMesh)scene.remove(droneMesh);
 propellers=[];
 const request=++droneModelRequest;
 droneMesh=new THREE.Group();
 const body=new THREE.Mesh(new THREE.BoxGeometry(.18,.07,.18),new THREE.MeshStandardMaterial({color:0x263e4a,metalness:.6,roughness:.35}));
 droneMesh.add(body);
 for(let i=0;i<4;i++){
  const p=profile.rotors[i].position_m,x=p[0],z=-p[1];
  const arm=new THREE.Mesh(new THREE.CylinderGeometry(.012,.012,Math.hypot(x,z),10),new THREE.MeshStandardMaterial({color:0x6a8998,metalness:.5}));
  arm.position.set(x/2,0,z/2);arm.rotation.z=-Math.atan2(x,z);droneMesh.add(arm);
  const motor=new THREE.Mesh(new THREE.CylinderGeometry(.025,.025,.025,16),new THREE.MeshStandardMaterial({color:0x56d6a0,metalness:.6}));
  motor.position.set(x,.025,z);droneMesh.add(motor);
  const prop=new THREE.Mesh(new THREE.BoxGeometry(.12,.006,.012),new THREE.MeshStandardMaterial({color:0x9db8c2,transparent:true,opacity:.72}));
  prop.position.set(x,.043,z);droneMesh.add(prop);propellers.push(prop);
 }
 scene.add(droneMesh);
 if(!THREE.GLTFLoader){diagLog("warn","GLTFLoader unavailable; using the profile-based fallback model","cad.glb.loader");return;}
 new THREE.GLTFLoader().load("./models/current/drone.glb",gltf=>{
  if(request!==droneModelRequest||!droneMesh)return;
  const model=gltf.scene,byRotor=new Array(4);
  model.traverse(node=>{
   if(!node.isMesh||!/^prop_[1-4]$/.test(node.name))return;
   let best=-1,bestDistance=Infinity;
   profile.rotors.forEach((rotor,index)=>{
    const dx=node.position.x-Number(rotor.position_m[0]);
    const dz=node.position.z+Number(rotor.position_m[1]);
    const distance=dx*dx+dz*dz;
    if(distance<bestDistance){bestDistance=distance;best=index;}
   });
   if(best>=0&&bestDistance<1e-6)byRotor[best]=node;
  });
  while(droneMesh.children.length)droneMesh.remove(droneMesh.children[0]);
  droneMesh.add(model);
  propellers=byRotor;
  $("profileCaption").textContent=profile.display_name+" · CAD Smooth2/kilo";
  if(propellers.some(p=>!p))diagLog("warn","GLB loaded but one or more propeller nodes could not be mapped to simulator rotors","cad.glb.rotor-map");
  else diagLog("ok","26-part Smooth2 CAD GLB loaded; propellers mapped to the active rotor profile","cad.glb.loaded");
 },undefined,error=>{
  if(request===droneModelRequest)diagLog("warn","CAD GLB load failed; using profile-based fallback: "+(error.message||String(error)),"cad.glb.load");
 });
}
function init3D(){const host=$("scene");scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(45,host.clientWidth/host.clientHeight,.1,100);camera.position.set(2.7,2.4,3.7);camera.lookAt(0,.65,0);renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(host.clientWidth,host.clientHeight);host.appendChild(renderer.domElement);scene.add(new THREE.HemisphereLight(0xb9e5ff,0x20313b,2));const light=new THREE.DirectionalLight(0xffffff,2.2);light.position.set(4,8,5);scene.add(light);const grid=new THREE.GridHelper(10,20,0x315365,0x203b4c);scene.add(grid);buildDrone();new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h)}).observe(host);requestAnimationFrame(draw)}
function draw(now){requestAnimationFrame(draw);if(!lastFrame)lastFrame=now;const elapsed=Math.min((now-lastFrame)/1000,.1);lastFrame=now;
 if(armed&&!paused&&pyodide&&!stepBusy){accumulator+=elapsed;let count=0;while(accumulator>=DT&&count<stepsPerFrame){if(keys.w||keys.s||keys.arrowup||keys.arrowdown||keys.arrowleft||keys.arrowright||keys.a||keys.d)directMotorMode=false;if(keys.w)throttle=clamp(throttle+.55*DT,0,1);if(keys.s)throttle=clamp(throttle-.55*DT,0,1);$("throttle").value=throttle*100;$("throttleValue").textContent=Math.round(throttle*100)+"%";const cmd=currentAction();motors=cmd;try{snapshot=JSON.parse(engineStep(JSON.stringify(cmd)));if(snapshot.step%50===0)renderEngineDiagnostics();}catch(e){diagLog("error",e.stack||e.message,"physics.step");log("خطای موتور فیزیک: "+e.message);armed=false;setStatus("خطا در موتور");break}accumulator-=DT;count++;if(snapshot.terminated||snapshot.truncated){armed=false;$("armBtn").textContent="مسلح‌سازی موتورها";log(snapshot.info?.crash_reasons?.join(", ")||"پایان اپیزود؛ برای شروع دوباره بازنشانی کنید.");break}}if(count)updateUI()}
 if(droneMesh&&snapshot){const s=snapshot.state;droneMesh.position.set(s[0],s[2],-s[1]);const m=threeRotation(s.slice(6,9));const mat=new THREE.Matrix4().set(m[0][0],m[0][1],m[0][2],0,m[1][0],m[1][1],m[1][2],0,m[2][0],m[2][1],m[2][2],0,0,0,0,1);droneMesh.quaternion.setFromRotationMatrix(mat)}
 propellers.forEach((p,i)=>{if(p)p.rotation.y+=motors[i]*.8*-Number(profile.rotors[i].spin_sign||1)});renderer.render(scene,camera)}
function currentAction(){if(directMotorMode)return motors.map(v=>clamp(v,0,1));let roll=(keys.arrowleft?1:0)-(keys.arrowright?1:0),pitch=(keys.arrowup?1:0)-(keys.arrowdown?1:0),yaw=(keys.d?1:0)-(keys.a?1:0);return [throttle+roll*.14-pitch*.14+yaw*.07,throttle+roll*.14-pitch*.14-yaw*.07,throttle-roll*.14+pitch*.14+yaw*.07,throttle-roll*.14+pitch*.14-yaw*.07].map(v=>clamp(v,0,1))}
function updateUI(){if(!snapshot)return;const s=snapshot.state,[x,y,z,vx,vy,vz,r,p,yaw]=s,speed=norm([vx,vy,vz]);$("alt").textContent=z.toFixed(3)+" m";$("vz").textContent=vz.toFixed(3)+" m/s";$("speed").textContent=speed.toFixed(3)+" m/s";$("thrust").textContent=Number(snapshot.total_thrust_n||0).toFixed(2)+" N";$("roll").textContent=(r*180/Math.PI).toFixed(1)+"°";$("pitch").textContent=(p*180/Math.PI).toFixed(1)+"°";$("yaw").textContent=(yaw*180/Math.PI).toFixed(1)+"°";$("mass").textContent=Number(snapshot.mass_kg).toFixed(3)+" kg";$("altHud").textContent=z.toFixed(2)+" m";$("speedHud").textContent=speed.toFixed(2)+" m/s";$("timeHud").textContent=Number(snapshot.time_s).toFixed(2)+" s";$("flightState").textContent=z>.18?"در حال پرواز":armed?"موتورها مسلح":"روی زمین";const actual=snapshot.info?.rotor_omega_rad_s||[0,0,0,0],max=profile.propulsion.max_rotor_speed_rad_s;$("motorBars").innerHTML=motors.map((v,i)=>{const pct=Math.round(v*100);return '<div class="bar-row"><span>موتور '+(i+1)+'</span><div class="bar"><i style="width:'+pct+'%"></i></div><b>'+pct+'%</b></div>'}).join("");$("statusText").textContent=armed?"موتور فیزیک آموزشی فعال":"موتور آماده";$("statusDot").style.background=armed?"#56d6a0":"#f0b66a";$("timeHud").textContent=Number(snapshot.time_s).toFixed(2)+" s";if(snapshot.step%25===0){$("engineMeta").textContent="گام "+snapshot.step+" · dt="+snapshot.dt_s+"s · obs="+snapshot.observation_size+" · reward="+Number(snapshot.reward||0).toFixed(3)}}
function loadExternalScript(src){return new Promise((resolve,reject)=>{const existing=[...document.scripts].find(s=>s.src===src);if(existing){if(window.loadPyodide)return resolve();existing.addEventListener("load",resolve,{once:true});existing.addEventListener("error",()=>reject(Error("بارگذاری اسکریپت ناموفق: "+src)),{once:true});return}const script=document.createElement("script");script.src=src;script.async=true;script.onload=resolve;script.onerror=()=>reject(Error("دریافت اسکریپت Pyodide ناموفق: "+src));document.head.appendChild(script)})}
async function probePyodideMirror(indexURL){
 const names=["pyodide.js","pyodide.asm.js","pyodide.asm.wasm","pyodide.asm.data","pyodide-lock.json"];
 const results=[];
 for(const name of names){
  const url=indexURL+name;
  try{
   const response=await fetch(url,{method:"HEAD",cache:"no-store"});
   results.push({file:name,status:response.status,ok:response.ok,type:response.headers.get("content-type"),length:response.headers.get("content-length")});
  }catch(e){results.push({file:name,error:e.message||String(e)})}
 }
 return results;
}
async function loadTrainingEngine(){
 let phase="شروع";
 const bootTrace=[];
 try{
  phase="بررسی قابلیت‌های مرورگر";
  const caps={wasm:typeof WebAssembly!=="undefined",fetch:typeof fetch==="function",secureContext:window.isSecureContext,crossOriginIsolated:window.crossOriginIsolated===true,sharedArrayBuffer:typeof SharedArrayBuffer!=="undefined",userAgent:navigator.userAgent};
  diagLog("info",JSON.stringify(caps),"pyodide.browser");
  if(!caps.wasm||!caps.fetch)throw Error("مرورگر WebAssembly یا Fetch را پشتیبانی نمی‌کند");
  phase="بارگذاری Pyodide";
  setStatus("در حال بارگذاری Python / Pyodide");log("در حال آماده‌سازی همان بستهٔ فیزیک مورد استفاده در PPO…");
  // Some Android/ARM browsers fail to initialize CPython on Pyodide 0.26+.
  // Use the 0.25.1 runtime reported to work on affected Android devices.
  const mirrors=[
   {script:"https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js",index:"https://cdn.jsdelivr.net/pyodide/v0.25.1/full/"},
   {script:"https://unpkg.com/pyodide@0.25.1/pyodide.js",index:"https://unpkg.com/pyodide@0.25.1/"}
  ];
  let bootErrors=[];
  for(const mirror of mirrors){
   const stderr=[];
   try{
    bootTrace.push("Probe "+mirror.index);
    const probe=await probePyodideMirror(mirror.index);
    bootTrace.push(JSON.stringify(probe));
    if(typeof window.loadPyodide!=="function")await loadExternalScript(mirror.script);
    if(typeof window.loadPyodide!=="function")throw Error("اسکریپت دریافت شد اما loadPyodide در دسترس نیست");
    pyodide=await window.loadPyodide({indexURL:mirror.index,stderr:(line)=>{stderr.push(String(line));console.error("[Pyodide stderr]",line)}});
    bootTrace.push("Pyodide initialized from "+mirror.index);
    break;
   }catch(bootError){
    bootErrors.push(mirror.index+" => "+(bootError?.stack||bootError?.message||String(bootError))+(stderr.length?" | stderr: "+stderr.join(" || "):""));
    pyodide=null;
   }
  }
  if(!pyodide)throw Error("راه‌اندازی Pyodide ناموفق بود. "+bootErrors.join(" || ")+" | diagnostics: "+bootTrace.join(" || "));
  phase="بارگذاری micropip";await pyodide.loadPackage(["micropip"]);
  phase="نصب Gymnasium";await pyodide.runPythonAsync("import micropip\nawait micropip.install('gymnasium==1.2.3')");
  const paths=["__init__.py","dynamics.py","parameters.py","propeller.py","system_identification.py","mass_properties.py","curriculum.py","env.py","vehicle_family.py","diagnostics.py"];
  pyodide.FS.mkdirTree("/repo/training/fast_physics");pyodide.FS.mkdirTree("/repo/vehicle_models/family");pyodide.FS.mkdirTree("/repo");
  phase="دریافت فایل‌های فیزیک";
  for(const name of paths){const url="./training/fast_physics/"+name;const res=await fetch(url);if(!res.ok)throw Error("بارگذاری "+url+" ناموفق: HTTP "+res.status);pyodide.FS.writeFile("/repo/training/fast_physics/"+name,await res.text())}
  phase="دریافت پروفایل‌های وسیله";
  const prof=await fetch("./vehicle_models/family/vehicle_profiles.json");if(!prof.ok)throw Error("فایل vehicle_profiles.json ناموفق: HTTP "+prof.status);
  pyodide.FS.writeFile("/repo/vehicle_models/family/vehicle_profiles.json",await prof.text());
  phase="دریافت تنظیمات تصادفی‌سازی";
  const rand=await fetch("./vehicle_models/family/domain_randomization.json");if(!rand.ok)throw Error("فایل domain_randomization.json ناموفق: HTTP "+rand.status);
  pyodide.FS.writeFile("/repo/vehicle_models/family/domain_randomization.json",await rand.text());
  phase="دریافت پل Python";
  const bridge=await fetch("./engine.py");if(!bridge.ok)throw Error("فایل bridge موتور ناموفق: HTTP "+bridge.status);
  pyodide.FS.writeFile("/repo/browser_engine.py",await bridge.text());
  phase="واردکردن موتور آموزشی";pyodide.runPython("import sys; sys.path.insert(0, '/repo'); import browser_engine; from browser_engine import engine_reset, engine_step, engine_diagnostics");
  engineReset=pyodide.globals.get("engine_reset");engineStep=pyodide.globals.get("engine_step");engineDiagnostics=pyodide.globals.get("engine_diagnostics");
  if(typeof engineReset!=="function"||typeof engineStep!=="function"||typeof engineDiagnostics!=="function")throw Error("توابع موتور از ماژول browser_engine به فضای JavaScript منتقل نشدند؛ reset="+typeof engineReset+", step="+typeof engineStep+", diagnostics="+typeof engineDiagnostics);
  phase="ساخت محیط و بازنشانی اولیه";
  const seed=Number($("seedInput").value)||12345,stage=Number($("curriculumSelect").value),randomize=$("randomizeToggle").checked;
  snapshot=JSON.parse(engineReset(profile.vehicle_id,randomize,stage,seed));
  diagLog("ok","محیط اصلی VehicleFamilyEnv و FastQuadrotor ساخته و reset شد","engine.ready");renderEngineDiagnostics();
  setStatus("موتور آموزشی بارگذاری شد",true)
 }catch(e){const detail=(e&&e.stack)?e.stack:(e&&e.message?e.message:String(e));diagLog("error",detail,"startup."+phase);throw new Error("مرحلهٔ "+phase+"؛ "+detail)}
}
function resetEngine(){if(!engineReset)return;armed=false;paused=false;motors=[0,0,0,0];directMotorMode=false;throttle=0;$("throttle").value=0;$("throttleValue").textContent="0%";const seed=Number($("seedInput").value)||12345,stage=Number($("curriculumSelect").value),randomize=$("randomizeToggle").checked;const raw=engineReset(profile.vehicle_id,randomize,stage,seed);snapshot=JSON.parse(raw);renderEngineDiagnostics();$("armBtn").textContent="مسلح‌سازی موتورها";$("pauseBtn").textContent="توقف";$("flightState").textContent="روی زمین";$("profileCaption").textContent=profile.display_name+" · "+profile.calibration_status;$("modeChip").textContent="VehicleFamilyEnv / FastQuadrotor";$("engineMeta").textContent="dt=0.02s · obs="+snapshot.observation_size+" · action=4 · seed="+seed;log("محیط واقعی آموزش بازنشانی شد. سناریو: "+JSON.stringify(snapshot.info.scenario));updateUI()}
function buildMotorSliders(){$("motorGrid").innerHTML=[0,1,2,3].map(i=>'<label>موتور '+(i+1)+'<b id="mval'+i+'">0%</b><input type="range" min="0" max="100" value="0" id="m'+i+'"></label>').join("");for(let i=0;i<4;i++)$("m"+i).addEventListener("input",e=>{motors[i]=Number(e.target.value)/100;directMotorMode=true;$("mval"+i).textContent=e.target.value+"%";if(armed)log("فرمان مستقیم روتورها؛ محیط همان VehicleFamilyEnv است.")})}
async function init(){try{const response=await fetch("./vehicle_models/family/vehicle_profiles.json");if(!response.ok)throw Error("HTTP "+response.status);profiles=(await response.json()).profiles;profile=profiles[0];$("profileSelect").innerHTML=profiles.map((p,i)=>'<option value="'+i+'">'+p.display_name+'</option>').join("");$("profileSelect").addEventListener("change",()=>{profile=profiles[Number($("profileSelect").value)];buildDrone();resetEngine()});buildMotorSliders();init3D();await loadTrainingEngine();resetEngine();}catch(e){console.error("Flight Lab startup failure",e);diagLog("error",e.stack||e.message,"startup.failure");setStatus("خطا در بارگذاری موتور");log("خطای دقیق راه‌اندازی: "+(e.message||String(e)))}}
$("armBtn").addEventListener("click",()=>{if(!snapshot||snapshot.terminated||snapshot.truncated)return;armed=!armed;$("armBtn").textContent=armed?"قطع موتورها":"مسلح‌سازی موتورها";if(!armed)motors=[0,0,0,0];setStatus(armed?"موتور در حال اجرا":"موتور متوقف شد",true)});
$("copyDiagnostics").addEventListener("click",copyAllLogs);$("clearDiagnostics").addEventListener("click",()=>{diagnosticHistory=[];const panel=$("diagnosticLog");if(panel)panel.textContent="نمایش لاگ پاک شد؛ گزارش بعدی از این نقطه ثبت می‌شود."});$("resetBtn").addEventListener("click",resetEngine);$("pauseBtn").addEventListener("click",()=>{paused=!paused;$("pauseBtn").textContent=paused?"ادامه":"توقف"});
$("throttle").addEventListener("input",e=>{throttle=Number(e.target.value)/100;directMotorMode=false;$("throttleValue").textContent=e.target.value+"%"});
$("profileSelect").addEventListener("change",()=>{});$("randomizeToggle").addEventListener("change",()=>resetEngine());$("curriculumSelect").addEventListener("change",()=>resetEngine());$("seedInput").addEventListener("change",()=>resetEngine());
window.addEventListener("keydown",e=>{const k=e.key.toLowerCase();if(["arrowup","arrowdown","arrowleft","arrowright"," "].includes(k))e.preventDefault();keys[k]=true;if(e.code==="Space"){armed=false;$("armBtn").textContent="مسلح‌سازی موتورها"}});window.addEventListener("keyup",e=>keys[e.key.toLowerCase()]=false);
init();
