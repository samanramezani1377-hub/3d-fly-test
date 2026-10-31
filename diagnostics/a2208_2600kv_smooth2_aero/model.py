#!/usr/bin/env python3
"""Diagnostic static BEMT + DC motor/prop operating-point solver.

This is a transparent engineering estimate, not CFD and not a measured thrust curve.
The default section polar is explicitly a rough placeholder. Replace it with measured
or validated low-Reynolds-number polar CSV data before calling the result high-confidence.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RHO, MU, G = 1.225, 1.81e-5, 9.80665
GEOM = json.loads((HERE / "smooth2_geometry.json").read_text())
MOTOR = json.loads((HERE / "a2208_motor.json").read_text())

def clamp(x, lo, hi): return max(lo, min(hi, x))

def polar_placeholder(alpha_rad, reynolds):
    # Deliberately conservative/simple placeholder; not a NACA 4410 measured polar.
    alpha_deg = math.degrees(alpha_rad)
    cl = clamp(2 * math.pi * (alpha_rad - math.radians(-2.0)), -1.0, 1.0)
    # Penalize low-Re operation; this is a sensitivity approximation, not validated data.
    re_factor = clamp((reynolds / 30000.0) ** 0.18, 0.48, 1.0)
    cl *= re_factor
    cd = 0.025 + 0.018 * cl * cl + 0.12 / math.sqrt(max(reynolds, 500.0))
    return cl, cd

def load_polar(path):
    rows=[]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["alpha_deg"]), float(row["cl"]), float(row["cd"])))
    if len(rows) < 5: raise ValueError("Polar CSV must contain at least 5 alpha/CL/CD rows")
    return sorted(rows)

def polar_from_table(alpha_rad, reynolds, tables):
    # Tables: [(Re, [(alpha_deg, cl, cd), ...]), ...]; interpolate alpha then Re.
    def at_re(table):
        a=math.degrees(alpha_rad); rows=table[1]
        if a <= rows[0][0]: return rows[0][1], rows[0][2]
        if a >= rows[-1][0]: return rows[-1][1], rows[-1][2]
        for p,q in zip(rows,rows[1:]):
            if p[0] <= a <= q[0]:
                t=(a-p[0])/(q[0]-p[0])
                return p[1]+t*(q[1]-p[1]), p[2]+t*(q[2]-p[2])
    tables=sorted(tables,key=lambda x:x[0])
    if reynolds <= tables[0][0]: return at_re(tables[0])
    if reynolds >= tables[-1][0]: return at_re(tables[-1])
    for lo,hi in zip(tables,tables[1:]):
        if lo[0] <= reynolds <= hi[0]:
            t=(reynolds-lo[0])/(hi[0]-lo[0]); a=at_re(lo); b=at_re(hi)
            return a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])

def prop_load(rpm, polar=None, sections=36):
    scale=GEOM["model_scale"]
    R=GEOM["prop_radius_mm"]*scale/1000
    pitch=GEOM["pitch_mm"]/1000
    blades=GEOM["blades"]; omega=rpm*2*math.pi/60
    stations=GEOM["stations"]
    def interp_chord(frac):
        for (x0,c0),(x1,c1) in zip(stations,stations[1:]):
            if x0 <= frac <= x1:
                t=(frac-x0)/(x1-x0); return (c0+t*(c1-c0))*scale/1000
        return stations[-1][1]*scale/1000
    thrust=torque=0.0; re_vals=[]
    # BEMT annuli; solve induced axial velocity with local momentum balance.
    for j in range(sections):
        r=R*(0.06+0.93*(j+0.5)/sections)
        dr=R*0.93/sections
        frac=r/R; chord=interp_chord(frac)
        beta=min(math.atan(pitch/(2*math.pi*r)), math.radians(42))
        def section(vi):
            vt=omega*r; w=math.hypot(vt,vi); phi=math.atan2(vi,vt)
            re=RHO*w*chord/MU
            cl,cd=polar(beta-phi,re) if polar else polar_placeholder(beta-phi,re)
            q=.5*RHO*w*w
            dL=q*chord*cl*blades
            dD=q*chord*cd*blades
            dT=dL*math.cos(phi)-dD*math.sin(phi)
            dQ=r*(dL*math.sin(phi)+dD*math.cos(phi))
            # Prandtl tip/root loss factor.
            s=max(abs(math.sin(phi)),1e-5)
            ftip=blades/2*(R-r)/(r*s)
            froot=blades/2*(r-0.06*R)/(0.06*R*s)
            F=clamp((2/math.pi)*math.acos(math.exp(-max(ftip,0)))*
                    (2/math.pi)*math.acos(math.exp(-max(froot,0))),0.05,1.0)
            momentum=4*math.pi*RHO*r*vi*vi*F
            return dT-momentum,dT*F,dQ*F,re
        lo,hi=0.0,100.0
        for _ in range(60):
            mid=(lo+hi)/2
            if section(mid)[0] > 0: lo=mid
            else: hi=mid
        _,dT,dQ,re=section((lo+hi)/2)
        thrust+=dT*dr; torque+=dQ*dr; re_vals.append(re)
    return thrust,torque,min(re_vals),max(re_vals)

def motor_torque(rpm, voltage, current_limit):
    kv=MOTOR["kv_rpm_per_volt"]
    kt=60/(2*math.pi*kv)  # Nm/A
    omega=rpm*2*math.pi/60
    ke=1/(kv*2*math.pi/60) # V/(rad/s)
    resistance=MOTOR["winding_resistance_ohm_reported"]
    i0=MOTOR["no_load_current_a_reported_at_10v"]
    # Approximate no-load current scaling around the published 10V point.
    i0_at_v=i0*voltage/10.0
    current=max(0.0,(voltage-ke*omega)/resistance)
    current=min(current,current_limit)
    return kt*max(0.0,current-i0_at_v),current

def solve_operating_point(cells, battery_sag_v=0.0, current_limit=None, polar=None):
    v=cells*3.7-battery_sag_v
    limit=current_limit or MOTOR["maximum_current_a_reported"]
    kv=MOTOR["kv_rpm_per_volt"]
    rpm_hi=kv*v*0.999
    def residual(rpm):
        tq,_=motor_torque(rpm,v,limit)
        _,load,_,_=prop_load(rpm,polar)
        return tq-load
    lo,hi=100.0,rpm_hi
    if residual(lo) <= 0: rpm=lo
    else:
        for _ in range(70):
            mid=(lo+hi)/2
            if residual(mid)>0: lo=mid
            else: hi=mid
        rpm=(lo+hi)/2
    thrust,load_torque,re_min,re_max=prop_load(rpm,polar)
    _,amps=motor_torque(rpm,v,limit)
    return {"cells":cells,"voltage_at_motor_V":v,"rpm":rpm,"thrust_N":thrust,
            "thrust_gf":thrust/G*1000,"prop_torque_Nm":load_torque,
            "current_A":amps,"electrical_input_W":v*amps,
            "Reynolds_min":re_min,"Reynolds_max":re_max}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cells",type=int,nargs="+",default=[3])
    ap.add_argument("--battery-sag-v",type=float,default=0.0)
    ap.add_argument("--current-limit-a",type=float,default=None)
    ap.add_argument("--polar-csv",type=Path)
    args=ap.parse_args()
    polar=None
    if args.polar_csv:
        rows=load_polar(args.polar_csv)
        polar=lambda alpha,re: polar_from_table(alpha,re,[(30000,rows)])
    results=[solve_operating_point(c,args.battery_sag_v,args.current_limit_a,polar) for c in args.cells]
    print(json.dumps({"model":"static BEMT + approximate DC motor equilibrium",
      "polar_source":"user CSV" if polar else "placeholder low-Re polar (low confidence)",
      "motor":MOTOR["motor"],"propeller_diameter_mm":2*GEOM["prop_radius_mm"]*GEOM["model_scale"],
      "results":results,
      "limitations":["No CFD or measured propeller polar is bundled.",
        "Reported motor data may vary by A2208/8T manufacturer/clone.",
        "Battery sag, ESC losses and motor heating are not inferred unless supplied.",
        "Static thrust only; no guard interference, ground effect or forward-flight inflow.",
        "Do not treat this diagnostic as flight calibration or safety approval."]},indent=2))
if __name__=="__main__": main()
