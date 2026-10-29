#!/usr/bin/env python3
"""Prepare a pinned SkySim checkout using this repo's editable drone config/controller."""
import argparse
import json
import re
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = args.upstream.resolve()
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
    if min(mass, rotor_radius, arm_length, motor_kv, voltage, ground_radius, model_scale) <= 0:
        raise ValueError("Mass, rotor radius, arm length, motor KV, voltage, ground radius and model scale must be positive")
    if any(float(inertia[k]) <= 0 for k in ("x", "y", "z")):
        raise ValueError("All inertia values must be positive")

    header_path = upstream / "web/core/drone_core.hpp"
    core = header_path.read_text(encoding="utf-8")
    core = replace_once(core, r"_rotor_radius\{[^}]+\}", f"_rotor_radius{{{rotor_radius}}}", "rotor radius")
    core = replace_once(core, r"_motor_kv\{[^}]+\}, _max_voltage\{[^}]+\}", f"_motor_kv{{{motor_kv}}}, _max_voltage{{{voltage}}}", "motor KV and max voltage")
    core = replace_once(core, r"_mass\{[^}]+\}", f"_mass{{{mass}}}", "mass")
    core = replace_once(core, r"_inertia\{[^}]+\}", f"_inertia{{{float(inertia['x'])}, {float(inertia['y'])}, {float(inertia['z'])}}}", "inertia")
    core = replace_once(core, r"_ground_radius\{[^}]+\}", f"_ground_radius{{{ground_radius}}}", "ground contact radius")
    header_path.write_text(core, encoding="utf-8")

    cpp_path = upstream / "web/core/drone_core.cpp"
    cpp = cpp_path.read_text(encoding="utf-8")
    cpp = replace_once(cpp, r"const double arm = _rotor_radius \* 2\.1;", f"const double arm = {arm_length};", "motor arm length")
    cpp_path.write_text(cpp, encoding="utf-8")

    index_path = upstream / "web/index.html"
    html = index_path.read_text(encoding="utf-8")
    html = replace_once(
        html,
        r'<script src="https://cdnjs\\.cloudflare\\.com/ajax/libs/three\\.js/r128/three\\.min\\.js"></script>',
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\\n<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>',
        "GLTF loader",
    )
    html = replace_once(
        html,
        r'<script src="skysim_core\\.js"',
        '<script src="drone-config.js"></script>\\n<script src="skysim_core.js"',
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
    (upstream / "web/drone-config.js").write_text(
        "window.DRONE_CONFIG = " + json.dumps(browser_config, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print("SkySim prepared with config/drone.json and controllers/controller.py")
    print("GLB visual:", model_url or "not supplied; using generic quad visual")


if __name__ == "__main__":
    main()
