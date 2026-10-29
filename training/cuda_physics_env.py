"""CUDA-only hover task backed by the four-rotor physics prototype."""
from __future__ import annotations
import torch
from cuda_quadrotor_physics import CudaQuadrotorPhysics


class CudaPhysicsHoverBatch:
    def __init__(self, n_envs, device="cuda", dt=0.02, max_steps=1500,
                 target=(0.0, 3.0, 0.0), mass=1.5, arm=0.2667,
                 inertia=(0.02, 0.035, 0.02), rotor_radius=0.127,
                 battery_mah=4000, battery_resistance=0.03, cells=4, max_voltage=14.8):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; refusing CPU fallback.")
        self.n, self.dt, self.max_steps = int(n_envs), float(dt), int(max_steps)
        self.target = torch.tensor(target, dtype=torch.float32, device=self.device).view(1,3)
        self.physics = CudaQuadrotorPhysics(self.n, str(self.device), self.dt, mass, arm,
                                             inertia, rotor_radius, battery_mah,
                                             battery_resistance, cells, max_voltage)
        self.steps = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.took_off = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.ground_wait = torch.zeros(self.n, device=self.device)
        self.wind = torch.zeros((self.n,3), device=self.device)

    def observation(self):
        p,v,e,w = self.physics.position,self.physics.velocity,self.physics.euler(),self.physics.omega
        return torch.cat((p,v,e,w),-1)

    def reset(self, mask=None):
        if mask is None: mask = torch.ones(self.n,dtype=torch.bool,device=self.device)
        self.physics.reset(mask)
        self.steps = torch.where(mask,torch.zeros_like(self.steps),self.steps)
        self.took_off = torch.where(mask,torch.zeros_like(self.took_off),self.took_off)
        self.ground_wait = torch.where(mask,torch.zeros_like(self.ground_wait),self.ground_wait)
        return self.observation()

    def step(self, action):
        p,v,e,w = self.physics.step(action,self.wind)
        self.steps += 1
        newly = (~self.took_off) & (p[:,1]>=0.45)
        self.took_off |= newly
        self.ground_wait = torch.where(self.took_off,torch.zeros_like(self.ground_wait),self.ground_wait+1)
        err=p-self.target
        alt=torch.exp(-err[:,1].abs()/0.75)
        horiz=torch.linalg.vector_norm(err[:,[0,2]],dim=-1)
        hscore=torch.exp(-horiz)
        reward=2*(alt-0.5)+0.5*(hscore-0.5)
        reward-=0.12*v[:,1].square().clamp(max=16)
        reward-=0.04*v[:,[0,2]].square().sum(-1).clamp(max=25)
        reward-=0.08*e.square().sum(-1).clamp(max=9)
        reward-=0.03*w.square().sum(-1).clamp(max=25)
        reward-=0.01*action[:,:3].square().sum(-1)
        throttle=(0.35+0.15*action[:,3]).clamp(0,1)
        reward-=0.05*(throttle-0.35).square()
        reward-=0.01*self.ground_wait.clamp(max=50)*(~self.took_off)
        crashed=(self.took_off & (p[:,1]<=0.22)) | (p[:,1]>12) | (torch.linalg.vector_norm(p[:,[0,2]],dim=-1)>12) | (self.physics.soc<=0)
        reward-=100*crashed.float()
        done=crashed | (self.steps>=self.max_steps)
        # All reset operations remain on CUDA; avoid Python bool()/item() sync.
        self.reset(done)
        return self.observation(),reward,done,{"crashed":crashed}
