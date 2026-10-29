#!/usr/bin/env python3
"""Prepare a pinned SkySim checkout using this repo's editable drone config/controller."""
import argparse
import json
import re
import hashlib
import shutil
from pathlib import Path


def replace_once(text: str, pattern: str, replacement: str, label: str, flags=0) -> str:
    updated, count = re.subn(pattern, lambda _m: replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match for {label}; found {count}. Upstream source may have changed.")
    return updated


def num(cfg, key):
    value = cfg[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--commit-sha", default="unknown")
    parser.add_argument("--upstream-ref", default="a31864f999418531775980b879979ce4c57e791d")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = args.upstream.resolve()
    battery_source = root / "sim_support/battery.hpp"
    battery_target = upstream / "include/power/battery.hpp"
    battery_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(battery_source, battery_target)
    config = json.loads((root / "config/drone.json").read_text(encoding="utf-8"))
    controller = (root / "controllers/controller.py").read_text(encoding="utf-8").strip()
    if not controller or "def policy(obs)" not in controller:
        raise ValueError("controllers/controller.py must define policy(obs)")

    mass = num(config, "mass_kg")
    rotor_radius = num(config, "rotor_radius_m")
    arm_length = num(config, "arm_length_m")
    motor_kv = num(config, "motor_kv_rpm_per_volt")
    voltage = num(config, "max_voltage_v")
    ground_radius = num(config, "ground_contact_radius_m")
    inertia = config["inertia_kg_m2"]
    wind = config["wind_mps"]
    visual = config["visual"]
    model_scale = num(config, "model_scale")
    battery_cells = int(config.get("battery_cells", 4))
    battery_capacity_mah = num(config | {"battery_capacity_mah": config.get("battery_capacity_mah", 4000)}, "battery_capacity_mah")
    battery_resistance = num(config | {"battery_internal_resistance_ohm": config.get("battery_internal_resistance_ohm", 0.03)}, "battery_internal_resistance_ohm")
    if min(mass, rotor_radius, arm_length, motor_kv, voltage, ground_radius, model_scale) <= 0:
        raise ValueError("Mass, rotor radius, arm length, motor KV, voltage, ground radius and model scale must be positive")
    if battery_cells < 1 or battery_capacity_mah <= 0 or battery_resistance < 0:
        raise ValueError("Battery cells/capacity must be positive and internal resistance non-negative")
    if any(float(inertia[k]) <= 0 for k in ("x", "y", "z")):
        raise ValueError("All inertia values must be positive")

    header_path = upstream / "web/core/drone_core.hpp"
    core = header_path.read_text(encoding="utf-8")
    core = replace_once(core, r"_rotor_radius\{[^}]+\}", f"_rotor_radius{{{rotor_radius}}}", "rotor radius")
    core = replace_once(core, r"_motor_kv\{[^}]+\}, _max_voltage\{[^}]+\}", f"_motor_kv{{{motor_kv}}}, _max_voltage{{{voltage}}}", "motor KV and max voltage")
    core = replace_once(core, r"_mass\{[^}]+\}", f"_mass{{{mass}}}", "mass")
    core = replace_once(core, r"_inertia\{[^}]+\}", f"_inertia{{{float(inertia['x'])}, {float(inertia['y'])}, {float(inertia['z'])}}}", "inertia")
    core = replace_once(core, r"_ground_radius\{[^}]+\}", f"_ground_radius{{{ground_radius}}}", "ground contact radius")
    core = replace_once(core, r"DroneCore\(\) \{ build_rotors\(\); reset\(0, 2, 0\); \}", f"DroneCore() {{ build_rotors(); _battery.configure({battery_capacity_mah}, {battery_cells}, {battery_resistance}, _max_voltage); reset(0, 2, 0); }}", "battery configuration")
    header_path.write_text(core, encoding="utf-8")

    cpp_path = upstream / "web/core/drone_core.cpp"
    cpp = cpp_path.read_text(encoding="utf-8")
    cpp = replace_once(cpp, r"const double arm = _rotor_radius \* 2\.1;", f"const double arm = {arm_length};", "motor arm length")
    cpp_path.write_text(cpp, encoding="utf-8")

    index_path = upstream / "web/index.html"
    html = index_path.read_text(encoding="utf-8")
    html = replace_once(
        html,
        r'<script src="https://cdnjs\.cloudflare\.com/ajax/libs/three\.js/r128/three\.min\.js"></script>',
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>',
        "GLTF loader",
    )
    html = replace_once(
        html,
        r'<script src="skysim_core\.js"',
        '<script src="drone-config.js"></script>\n<script src="skysim_core.js"',
        "drone config script",
    )
    html = replace_once(
        html,
        r"function makeDrone\(\) \{\n  const g = new THREE\.Group\(\);",
        "function makeDrone() {\n  const g = new THREE.Group();\n  const cfg = window.DRONE_CONFIG || {};\n  const v = cfg.visual || {};",
        "visual model config",
    )
    html = replace_once(html, r"new THREE\.BoxGeometry\(0\.3,0\.09,0\.3\)", "new THREE.BoxGeometry(v.body_x_m || 0.3, v.body_y_m || 0.09, v.body_z_m || 0.3)", "visual body dimensions")
    html = replace_once(html, r"const arm = 0\.28;", "const arm = cfg.arm_length_m || 0.28;", "visual arm length")
    html = replace_once(html, r"new THREE\.CylinderGeometry\(0\.13,0\.13,0\.02,16\)", "new THREE.CylinderGeometry(v.propeller_radius_m || 0.13, v.propeller_radius_m || 0.13, 0.02, 16)", "visual propeller radius")
    html = replace_once(
        html,
        r"const drone = makeDrone\(\); scene\.add\(drone\);",
        """const drone = makeDrone(); scene.add(drone);
if (window.DRONE_CONFIG && window.DRONE_CONFIG.model_url && THREE.GLTFLoader) {
  new THREE.GLTFLoader().load(window.DRONE_CONFIG.model_url, (gltf) => {
    drone.clear();
    const model = gltf.scene;
    model.scale.setScalar(Number(window.DRONE_CONFIG.model_scale || 1));
    drone.add(model);
  }, undefined, (error) => {
    document.getElementById("err").textContent = "GLB visual load failed: " + error.message;
  });
}""",
        "optional GLB visual",
    )
    html = replace_once(
        html,
        r'yaw:\s+Module\.cwrap\("skysim_yaw","number",\["number"\]\),',
        'yaw:  Module.cwrap("skysim_yaw","number",["number"]),\n    setWind: Module.cwrap("skysim_set_wind",null,["number","number","number","number"]),',
        "wind API binding",
    )
    html = replace_once(
        html,
        r"api\.reset\(core, 0, 2, 0\);",
        "api.reset(core, 0, 2, 0);\n  const wind = (window.DRONE_CONFIG && window.DRONE_CONFIG.wind_mps) || {x:0,y:0,z:0};\n  api.setWind(core, Number(wind.x || 0), Number(wind.y || 0), Number(wind.z || 0));",
        "wind initialization",
    )
    html = replace_once(
        html,
        r'<textarea id="pycode">.*?</textarea>',
        '<textarea id="pycode">' + controller.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</textarea>",
        "default Python controller",
        flags=re.DOTALL,
    )

    # Add reproducible flight-test recording and JSON/CSV exports to the browser UI.
    html = replace_once(
        html,
        r'<div class="hint" id="err" style="color:#ff8f8f"></div>',
        '''<div class="hint" id="err" style="color:#ff8f8f"></div>
  <div style="margin-top:12px;border-top:1px solid #23405e;padding-top:10px">
    <div><b>Flight test recorder</b></div>
    <label for="scenarioName" class="hint">Scenario / test ID</label>
    <input id="scenarioName" value="baseline-hover" style="width:100%;background:#07131f;color:#cfe6ff;border:1px solid #23405e;border-radius:7px;padding:7px;font-family:inherit"/>
    <div class="row">
      <button id="recordStartBtn">Start recording</button>
      <button id="recordStopBtn" disabled>Stop</button>
    </div>
    <div class="row">
      <button id="exportJsonBtn" disabled>Download JSON</button>
      <button id="exportCsvBtn" disabled>Download CSV</button>
    </div>
    <div class="hint" id="recordStatus">Not recording. Samples are captured at 20 Hz while the page is running.</div>
  </div>''',
        "flight recorder controls",
    )
    # Fix upstream JS editor: compile function declarations as a function body, not a parenthesized expression.
    html = replace_once(
        html,
        r'userFn = new Function\("return \(" \+ \$\("code"\)\.value \+ "\\n; control\)"\)\(\);',
        'userFn = new Function($("code").value + "\\n; return control;")();',
        "JavaScript controller compilation",
    )

    html = replace_once(
        html,
        r'// ---------- main loop ----------',
        r'''// ---------- structured flight-test recorder ----------
const recorder = { active:false, startedAt:null, stoppedAt:null, scenario:"", samples:[], lastSampleAt:-Infinity, runId:null };
let lastCommand = [0,0,0,0];
function recorderStatus(message) {
  document.getElementById("recordStatus").textContent = message;
}
function recorderMeta() {
  const cfg = window.DRONE_CONFIG || {};
  return {
    schema_version: 1,
    run_id: recorder.runId,
    scenario: recorder.scenario,
    started_at_utc: recorder.startedAt,
    stopped_at_utc: recorder.stoppedAt,
    sample_count: recorder.samples.length,
    repository: "samanramezani1377-hub/3d-fly-test",
    build: cfg.build_info || {},
    simulator: "SkySim browser WebAssembly core",
    simulator_revision: (cfg.build_info || {}).upstream_ref || "unknown",
    controller_mode: mode,
    controller_source: cfg.controller_source || null,
    drone_config: cfg,
    recording_rate_hz: 20,
    notes: "Simulation telemetry only; visual GLB does not define physics. Not a real-flight safety validation."
  };
}
function startRecording() {
  recorder.active = true;
  recorder.startedAt = new Date().toISOString();
  recorder.stoppedAt = null;
  recorder.scenario = (document.getElementById("scenarioName").value || "unnamed-test").trim();
  recorder.samples = [];
  recorder.lastSampleAt = -Infinity;
  recorder.runId = (crypto.randomUUID ? crypto.randomUUID() : "run-" + Date.now() + "-" + Math.random().toString(16).slice(2));
  document.getElementById("recordStartBtn").disabled = true;
  document.getElementById("recordStopBtn").disabled = false;
  document.getElementById("exportJsonBtn").disabled = true;
  document.getElementById("exportCsvBtn").disabled = true;
  recorderStatus("Recording: " + recorder.scenario);
}
function stopRecording() {
  if (!recorder.active) return;
  recorder.active = false;
  recorder.stoppedAt = new Date().toISOString();
  document.getElementById("recordStartBtn").disabled = false;
  document.getElementById("recordStopBtn").disabled = true;
  document.getElementById("exportJsonBtn").disabled = recorder.samples.length === 0;
  document.getElementById("exportCsvBtn").disabled = recorder.samples.length === 0;
  recorderStatus("Stopped. " + recorder.samples.length + " samples captured. Download JSON for metadata + samples, or CSV for the time series.");
}
function captureSample(simTime) {
  if (!recorder.active || simTime - recorder.lastSampleAt < 0.05) return;
  recorder.lastSampleAt = simTime;
  const o = readObs();
  recorder.samples.push({
    timestamp_utc: new Date().toISOString(),
    sim_time_s: Number(simTime.toFixed(4)),
    armed: !!armed,
    controller_mode: mode,
    x_m:o.pos[0], y_m:o.pos[1], z_m:o.pos[2],
    vx_mps:o.vel[0], vy_mps:o.vel[1], vz_mps:o.vel[2],
    roll_rad:o.euler[0], pitch_rad:o.euler[1], yaw_rad:o.euler[2],
    wx_radps:o.rates[0], wy_radps:o.rates[1], wz_radps:o.rates[2],
    altitude_m:api.alt(core), vertical_speed_mps:api.vs(core),
    roll_deg:api.roll(core), pitch_deg:api.pitch(core), yaw_deg:api.yaw(core),
    thrust_n:api.thr(core),
    commanded_roll:lastCommand[0], commanded_pitch:lastCommand[1],
    commanded_yaw_rate:lastCommand[2], commanded_throttle:lastCommand[3]
  });
  if (recorder.samples.length % 100 === 0) recorderStatus("Recording: " + recorder.scenario + " · " + recorder.samples.length + " samples");
}
function downloadText(filename, content, mime) {
  const blob = new Blob([content], {type:mime});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href=url; a.download=filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function safeFilename() {
  return (recorder.scenario || "flight-test").replace(/[^a-z0-9_-]+/gi,"-").replace(/^-+|-+$/g,"") || "flight-test";
}
document.getElementById("recordStartBtn").addEventListener("click", startRecording);
document.getElementById("recordStopBtn").addEventListener("click", stopRecording);
document.getElementById("exportJsonBtn").addEventListener("click", () => {
  if (!recorder.samples.length) return;
  downloadText(safeFilename() + "-" + recorder.runId + ".json", JSON.stringify({...recorderMeta(), samples:recorder.samples}, null, 2), "application/json");
});
document.getElementById("exportCsvBtn").addEventListener("click", () => {
  if (!recorder.samples.length) return;
  const rows = recorder.samples, keys = Object.keys(rows[0]);
  const csv = [keys.join(","), ...rows.map(row => keys.map(k => {
    const value = row[k];
    return typeof value === "string" ? '"' + value.replace(/"/g,'""') + '"' : String(value);
  }).join(","))].join("\r\n");
  downloadText(safeFilename() + "-" + recorder.runId + ".csv", csv, "text/csv;charset=utf-8");
});

// ---------- main loop ----------''',
        "structured flight recorder",
    )
    html = replace_once(
        html,
        r'const \[r,p,yw,th\] = computeAction\(obs, now - t0\);\n      api\.setAtt\(core, r, p, yw, th\);',
        'const [r,p,yw,th] = computeAction(obs, now - t0);\n      lastCommand = [r,p,yw,th];\n      api.setAtt(core, r, p, yw, th);',
        "record commanded control outputs",
    )
    html = replace_once(
        html,
        r'api\.step\(core, DT\);\n    acc -= DT;',
        'api.step(core, DT);\n    captureSample(now - t0);\n    acc -= DT;',
        "record simulator telemetry",
    )

    index_path.write_text(html, encoding="utf-8")

    model_path = root / "models/current/drone.glb"
    model_url = ""
    if model_path.is_file():
        target = upstream / "web/current-drone.glb"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, target)
        model_url = "current-drone.glb"
    else:
        stale = upstream / "web/current-drone.glb"
        if stale.exists():
            stale.unlink()

    browser_config = dict(config)
    browser_config["model_url"] = model_url
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.is_file() else None
    browser_config["build_info"] = {"repository_commit": args.commit_sha, "upstream_repository": "vishwagw/Sky-Sim-drone-simulator-3.0", "upstream_ref": args.upstream_ref, "model_sha256": model_sha256}
    browser_config["controller_source"] = controller
    (upstream / "web/drone-config.js").write_text(
        "window.DRONE_CONFIG = " + json.dumps(browser_config, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print("SkySim prepared with config/drone.json and controllers/controller.py")
    print("GLB visual:", model_url or "not supplied; using generic quad visual")


if __name__ == "__main__":
    main()
