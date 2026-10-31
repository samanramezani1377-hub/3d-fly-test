"""Smooth2 CAD geometry and approximate static BEMT estimate (not measured)."""
import argparse, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAD = ROOT / "cad/source/smooth2_model_params.json"
PROFILE = ROOT / "vehicle_models/family/vehicle_profiles.json"
OUT = ROOT / "vehicle_models/family/smooth2_cad_propulsion_estimate.json"
CAD_BLOB_SHA = "cc77601fb0e4837ca1f7d9c40024c2bc942462"


def derive():
    p = json.loads(CAD.read_text())
    family = json.loads(PROFILE.read_text())
    v = next(x for x in family["profiles"] if x["vehicle_id"] == "smooth2-a2212-custom-prop")
    m, pr = float(p["model_scale"]), v["propulsion"]
    rho, blades = 1.225, 2
    R0 = float(p["prop_radius"]) / 1000
    R, D = R0*m, 2*R
    pitch = float(p["prop_pitch_mm"]) / 1000
    rpm = float(pr["kv_rpm_per_volt"]) * float(pr["nominal_battery_voltage_v"]) * .8
    omega = rpm*2*math.pi/60
    stations = p["prop_blade_stations"]
    rs = [float(s[0])*R0 for s in stations]
    cs = [float(s[1])*m/1000 for s in stations]
    T = Q = 0.0
    for i in range(len(rs)-1):
        r, dr = (rs[i]+rs[i+1])/2*m, (rs[i+1]-rs[i])*m
        c = (cs[i]+cs[i+1])/2
        beta = min(math.atan((pitch/m)/(2*math.pi*(r/m))), math.radians(42))
        def section(vi):
            vt=omega*r; W=math.hypot(vt,vi); phi=math.atan2(vi,vt)
            cl=max(-1.2,min(1.2,2*math.pi*(beta-phi)))
            cd=.012+.01*cl*cl
            dt=.5*rho*W*W*blades*c*(cl*math.cos(phi)-cd*math.sin(phi))
            dq=.5*rho*W*W*blades*c*r*(cd*math.cos(phi)+cl*math.sin(phi))
            return dt-4*math.pi*rho*r*vi*vi,dt,dq
        lo,hi=0.,80.
        for _ in range(64):
            mid=(lo+hi)/2
            if section(mid)[0]>0: lo=mid
            else: hi=mid
        _,dt,dq=section((lo+hi)/2); T+=dt*dr; Q+=dq*dr
    n=rpm/60
    CT=T/(rho*n*n*D**4); CQ=Q/(rho*n*n*D**5)
    kf, kq=T/omega**2, Q/omega**2
    total=4*T; mass=float(v["physical"]["mass_kg"])
    return {
      "schema_version":1,"vehicle_id":v["vehicle_id"],"calibration_status":"analytical_estimate_not_measured",
      "source":{"repository":"samanramezani1377-hub/3D","branch":"kilo","file":"cad/parametric/model.params.json","blob_sha":CAD_BLOB_SHA,"source_repository_modified":False},
      "propeller":{"blades":2,"radius_m":R,"diameter_m":D,"pitch_mm":pitch*1000,"pitch_diameter_ratio":pitch/D,
        "airfoil":{"NACA_parameterization":"4410","camber":p["prop_airfoil_camber"],"camber_position":p["prop_airfoil_camber_position"],"thickness_ratio":p["prop_airfoil_thickness_ratio"],
        "chord_stations_mm":[[s[0],s[1]*m] for s in stations]}},
      "motor":{"reference":pr["motor_name"],"KV":pr["kv_rpm_per_volt"],"voltage_V":pr["nominal_battery_voltage_v"],"cells":pr["battery_cells"],"capacity_Ah":pr["battery_capacity_ah"],
        "ideal_no_load_rpm":pr["kv_rpm_per_volt"]*pr["nominal_battery_voltage_v"],"80pct_loaded_rpm_estimate":rpm,
        "CAD_can_radius_mm":p["motor_radius"]*m,"CAD_can_height_mm":p["motor_height"]*m,"CAD_hub_radius_mm":p["motor_hub_radius"]*m,"CAD_hub_height_mm":p["motor_hub_height"]*m},
      "BEMT_static_estimate":{"air_density_kg_m3":rho,"rpm":rpm,"omega_rad_s":omega,"thrust_N_per_rotor":T,"torque_Nm_per_rotor":Q,"CT":CT,"CQ":CQ,
        "thrust_coefficient_N_per_rad_s2":kf,"torque_coefficient_Nm_per_rad_s2":kq,"torque_to_thrust_m":Q/T},
      "consistency":{"existing_profile_mass_kg":mass,"four_rotor_thrust_N":total,"estimated_TWR":total/(mass*9.80665),
        "active_profile_Kf":pr["thrust_coefficient_n_per_rad_s2"],"BEMT_to_active_Kf_ratio":kf/pr["thrust_coefficient_n_per_rad_s2"],
        "active_profile_max_omega_rad_s":pr["max_rotor_speed_rad_s"]},
      "limitations":["Approximate thin-airfoil lift slope with capped CL and simple drag polar; no measured low-Re airfoil polars.",
        "Motor CAD is a simplified envelope, not a winding/stator/rotor model.","No winding resistance, current, ESC limit, torque-speed curve, motor time constant, or motor/prop inertia.",
        "No material density or measured airframe mass/CG.","Static isolated-rotor estimate only; not a real thrust coefficient."],
      "active_vehicle_profile_modified":False,"physical_flight_eligible":False
    }


if __name__ == "__main__":
    a=argparse.ArgumentParser(); a.add_argument("--output",type=Path,default=OUT); a.add_argument("--check",action="store_true"); x=a.parse_args()
    report=derive()
    if x.check:
        assert .074 < report["propeller"]["diameter_m"] < .076 and len(report["propeller"]["airfoil"]["chord_stations_mm"])==11
        assert 0 < report["BEMT_static_estimate"]["thrust_coefficient_N_per_rad_s2"] < 1e-6
        assert report["consistency"]["estimated_TWR"] < 1
    x.output.parent.mkdir(parents=True,exist_ok=True); x.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
