"""GPU-resident quadrotor rigid-body + four-rotor physics prototype.

The equations are tensorized and run on CUDA. Parameters for rotor thrust and
torque are derived from hover balance and configurable ratios because the repo
does not contain the pinned DroneCore rotor/aerodynamic source. This is a
transparent physical prototype, not a numerically equivalent DroneCore port.
Validate coefficients against source/bench data before using it for real flight.
"""
from __future__ import annotations
import torch


class CudaQuadrotorPhysics:
    def __init__(self, n: int, device: str = "cuda", dt: float = 0.02,
                 mass: float = 1.5, arm: float = 0.2667,
                 inertia=(0.02, 0.035, 0.02), rotor_radius: float = 0.127,
                 battery_mah: float = 4000.0, battery_resistance: float = 0.03,
                 cells: int = 4, max_voltage: float = 14.8):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CudaQuadrotorPhysics requires an available NVIDIA CUDA GPU.")
        self.n, self.dt, self.mass, self.arm = n, dt, mass, arm
        self.g = 9.80665
        self.radius = rotor_radius
        self.inertia = torch.tensor(inertia, device=self.device).view(1, 3)
        self.capacity_mah = battery_mah
        self.resistance = battery_resistance
        self.cells = cells
        self.nominal_voltage = max_voltage
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def reset(self, mask):
        m3 = mask[:, None]
        if not hasattr(self, "position"):
            self.position = torch.zeros((self.n, 3), device=self.device)
            self.velocity = torch.zeros_like(self.position)
            self.q = torch.zeros((self.n, 4), device=self.device)
            self.q[:, 3] = 1.0  # xyzw
            self.omega = torch.zeros_like(self.position)
            self.rotor_thrust = torch.zeros((self.n, 4), device=self.device)
            self.motor_power = torch.zeros(self.n, device=self.device)
            self.soc = torch.ones(self.n, device=self.device)
            self.battery_voltage = torch.full((self.n,), self.nominal_voltage, device=self.device)
            self.current = torch.zeros(self.n, device=self.device)
        self.position = torch.where(m3, torch.zeros_like(self.position), self.position)
        self.position[:, 1] = torch.where(mask, torch.full_like(self.position[:, 1], 0.16), self.position[:, 1])
        self.velocity = torch.where(m3, torch.zeros_like(self.velocity), self.velocity)
        identity = torch.zeros_like(self.q); identity[:, 3] = 1
        self.q = torch.where(m3, identity, self.q)
        self.omega = torch.where(m3, torch.zeros_like(self.omega), self.omega)
        self.rotor_thrust = torch.where(mask[:, None], torch.zeros_like(self.rotor_thrust), self.rotor_thrust)
        self.soc = torch.where(mask, torch.ones_like(self.soc), self.soc)
        self.battery_voltage = torch.where(mask, torch.full_like(self.battery_voltage, self.nominal_voltage), self.battery_voltage)
        self.current = torch.where(mask, torch.zeros_like(self.current), self.current)

    @staticmethod
    def _qnorm(q):
        return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _qmul(a, b):
        ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
        return torch.stack((aw*bx+ax*bw+ay*bz-az*by,
                            aw*by-ax*bz+ay*bw+az*bx,
                            aw*bz+ax*by-ay*bx+az*bw,
                            aw*bw-ax*bx-ay*by-az*bz), -1)

    @staticmethod
    def _rotate(q, v):
        qv, qw = q[:, :3], q[:, 3:4]
        t = 2 * torch.linalg.cross(qv, v, dim=-1)
        return v + qw*t + torch.linalg.cross(qv, t, dim=-1)

    def euler(self):
        x,y,z,w = self.q.unbind(-1)
        roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        sinp = (2*(w*y-z*x)).clamp(-1,1)
        pitch = torch.asin(sinp)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return torch.stack((roll,pitch,yaw), -1)

    def step(self, action, wind=None):
        """Action [N,4]: desired roll/pitch, yaw rate, collective throttle."""
        a = action.clamp(-1, 1)
        if wind is None: wind = torch.zeros_like(self.velocity)
        eul = self.euler()
        roll_sp, pitch_sp = 0.55*a[:,0], 0.55*a[:,1]
        yaw_rate_sp = 3.14*a[:,2]
        err_roll = torch.atan2(torch.sin(roll_sp-eul[:,0]), torch.cos(roll_sp-eul[:,0]))
        err_pitch = torch.atan2(torch.sin(pitch_sp-eul[:,1]), torch.cos(pitch_sp-eul[:,1]))
        # Rate/attitude controller -> desired body moments. Gains are explicit
        # prototype parameters and require calibration against DroneCore.
        moments = torch.stack((0.075*err_roll - 0.018*self.omega[:,0],
                               0.075*err_pitch - 0.018*self.omega[:,1],
                               0.025*(yaw_rate_sp-self.omega[:,1]*0-self.omega[:,2])), -1)
        throttle = (0.35 + 0.15*a[:,3]).clamp(0.05, 0.60)
        total_thrust_cmd = self.mass*self.g*(throttle/0.35)
        # Quad-X mixer: alternating rotor directions; normalized thrust allocation.
        # Collective is distributed with bounded roll/pitch/yaw differentials.
        roll_delta = moments[:,0] / (2*self.arm)
        pitch_delta = moments[:,1] / (2*self.arm)
        yaw_delta = moments[:,2] / (4*0.015)
        base = total_thrust_cmd[:,None] / 4
        desired = torch.stack((base-roll_delta/2+pitch_delta/2-yaw_delta,
                               base-roll_delta/2-pitch_delta/2+yaw_delta,
                               base+roll_delta/2-pitch_delta/2-yaw_delta,
                               base+roll_delta/2+pitch_delta/2+yaw_delta), -1).clamp_min(0)
        # First-order motor/rotor response, no Python-side per-environment loop.
        tau_motor = 0.045
        motor_alpha = min(self.dt / tau_motor, 1.0)
        self.rotor_thrust += (desired-self.rotor_thrust)*motor_alpha
        rotor_sum = self.rotor_thrust.sum(-1)
        roll_torque = self.arm*(self.rotor_thrust[:,1]+self.rotor_thrust[:,3]-self.rotor_thrust[:,0]-self.rotor_thrust[:,2])
        pitch_torque = self.arm*(self.rotor_thrust[:,0]+self.rotor_thrust[:,1]-self.rotor_thrust[:,2]-self.rotor_thrust[:,3])
        yaw_torque = 0.015*(-self.rotor_thrust[:,0]+self.rotor_thrust[:,1]-self.rotor_thrust[:,2]+self.rotor_thrust[:,3])
        torque = torch.stack((roll_torque,pitch_torque,yaw_torque), -1)
        # Euler rigid-body rotational dynamics, including gyroscopic cross term.
        iw = self.inertia*self.omega
        gyro = torch.linalg.cross(self.omega, iw, dim=-1)
        ang_acc = (torque-gyro)/self.inertia
        self.omega.add_(ang_acc*self.dt).clamp_(-30,30)
        omega_q = torch.cat((self.omega, torch.zeros((self.n,1),device=self.device)), -1)
        self.q = self._qnorm(self.q + 0.5*self._qmul(self.q, omega_q)*self.dt)

        up = self._rotate(self.q, torch.tensor([0.,1.,0.],device=self.device).expand(self.n,3))
        # Simple ground-effect multiplier, bounded to avoid singularity.
        height = (self.position[:,1]-0.0).clamp_min(0.16)
        ground_effect = (1.0 + 0.12*(self.radius/(4*height)).square()).clamp(1.0,1.25)
        thrust_world = up*(rotor_sum*ground_effect)[:,None]
        rel_v = self.velocity-wind
        drag = -0.08*rel_v - 0.12*rel_v.abs()*rel_v
        accel = (thrust_world+drag)/self.mass
        accel[:,1] -= self.g
        # Coarse VRS loss term when descending rapidly under high thrust.
        vrs = ((-self.velocity[:,1]-1.0).clamp_min(0)/4).clamp(max=0.35)
        accel[:,1] -= vrs*thrust_world[:,1].clamp_min(0)/self.mass
        self.velocity.add_(accel*self.dt)
        self.position.add_(self.velocity*self.dt)
        contact = self.position[:,1] < 0.16
        self.position[:,1].clamp_(min=0.16)
        self.velocity[:,1] = torch.where(contact & (self.velocity[:,1]<0), torch.zeros_like(self.velocity[:,1]), self.velocity[:,1])
        # Battery: first-order LiPo OCV, internal resistance and SOC; all GPU tensors.
        self.motor_power = (rotor_sum.clamp_min(0).pow(1.5)*0.12)
        voc_cell = 3.3 + 0.9*self.soc.clamp(0,1)
        voc = voc_cell*self.cells
        disc = (voc.square()-4*self.resistance*self.motor_power).clamp_min(0)
        # Resistance is a Python scalar, so branch in Python rather than passing
        # a bool to torch.where (which requires a tensor condition).
        if self.resistance > 1e-9:
            amps = (voc-torch.sqrt(disc))/(2*self.resistance)
        else:
            amps = self.motor_power/voc.clamp_min(1e-6)
        self.current = amps.clamp_min(0)
        self.battery_voltage = (voc-self.current*self.resistance).clamp_min(0)
        self.soc = (self.soc-self.current*self.dt/(self.capacity_mah*3.6)).clamp(0,1)
        return self.position, self.velocity, self.euler(), self.omega
