"""Experimental fully batched CUDA hover-task simulator.

This simulator keeps environment state, dynamics, reward, reset and termination
on the selected torch device. It is a GPU-native surrogate, NOT a bitwise port
of DroneCore: rotor blade-element aerodynamics, controller/mixer, battery,
ground-effect and vortex-ring-state models are not yet equivalent. Do not
describe policies trained here as DroneCore- or real-aircraft-validated.
"""
from __future__ import annotations

import math
import torch


class CudaHoverBatch:
    """Vectorized 12-observation/4-action hover task; no per-step CPU copies."""

    def __init__(self, n_envs: int, device: str = "cuda", dt: float = 1 / 50,
                 max_steps: int = 1500, target=(0.0, 3.0, 0.0)):
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. Refusing silent CPU fallback.")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaHoverBatch requires a CUDA device; CPU is not a supported training fallback.")
        self.n = int(n_envs)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.target = torch.tensor(target, device=self.device, dtype=torch.float32).view(1, 3)
        self.mass = 1.25
        self.gravity = 9.80665
        self.hover_throttle = 0.35
        self.step_count = torch.zeros(self.n, device=self.device, dtype=torch.int64)
        self.has_taken_off = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.ground_wait = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        self.position = torch.zeros((self.n, 3), device=self.device)
        self.velocity = torch.zeros_like(self.position)
        self.attitude = torch.zeros_like(self.position)  # roll, pitch, yaw (rad)
        self.angular_velocity = torch.zeros_like(self.position)
        self.wind = torch.zeros_like(self.position)
        self.reset()

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.n, device=self.device, dtype=torch.bool)
        self.position[mask] = 0
        self.position[mask, 1] = 0.16
        self.velocity[mask] = 0
        self.attitude[mask] = 0
        self.angular_velocity[mask] = 0
        self.step_count[mask] = 0
        self.has_taken_off[mask] = False
        self.ground_wait[mask] = 0
        return self.observation()

    def observation(self):
        return torch.cat((self.position, self.velocity, self.attitude, self.angular_velocity), dim=-1)

    def step(self, action):
        # Action is PPO's symmetric [-1,1] vector.
        a = action.clamp(-1.0, 1.0)
        roll_cmd = a[:, 0] * 0.55
        pitch_cmd = a[:, 1] * 0.55
        yaw_rate_cmd = a[:, 2] * 3.14
        throttle = (0.35 + 0.15 * a[:, 3]).clamp(0.0, 1.0)

        # GPU-side attitude controller surrogate. Actual DroneCore controller/mixer
        # and rotor aerodynamics must replace this before claiming parity.
        roll_err = torch.atan2(torch.sin(roll_cmd - self.attitude[:, 0]),
                               torch.cos(roll_cmd - self.attitude[:, 0]))
        pitch_err = torch.atan2(torch.sin(pitch_cmd - self.attitude[:, 1]),
                                torch.cos(pitch_cmd - self.attitude[:, 1]))
        torque = torch.stack((7.0 * roll_err - 2.0 * self.angular_velocity[:, 0],
                              7.0 * pitch_err - 2.0 * self.angular_velocity[:, 1],
                              1.8 * (yaw_rate_cmd - self.angular_velocity[:, 2])), dim=-1)
        inertia = torch.tensor([0.025, 0.025, 0.045], device=self.device)
        angular_acc = torque / inertia
        self.angular_velocity.add_(angular_acc * self.dt)
        self.angular_velocity.clamp_(-20.0, 20.0)
        self.attitude.add_(self.angular_velocity * self.dt)
        self.attitude[:, :2].clamp_(-1.45, 1.45)
        self.attitude[:, 2] = torch.atan2(torch.sin(self.attitude[:, 2]), torch.cos(self.attitude[:, 2]))

        roll, pitch, yaw = self.attitude.unbind(-1)
        # Body thrust points along body +Y in the simulator's Y-up convention.
        # Direction is the Z-Y-X Euler rotation applied to body up.
        thrust_dir = torch.stack((
            torch.sin(yaw) * torch.sin(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll),
            torch.cos(pitch) * torch.cos(roll),
            -torch.cos(yaw) * torch.sin(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll),
        ), dim=-1)
        thrust = self.mass * self.gravity * (throttle / self.hover_throttle)
        force = thrust[:, None] * thrust_dir
        relative_velocity = self.velocity - self.wind
        drag = -0.08 * relative_velocity - 0.12 * relative_velocity.abs() * relative_velocity
        acceleration = (force + drag) / self.mass
        acceleration[:, 1] -= self.gravity
        self.velocity.add_(acceleration * self.dt)
        self.position.add_(self.velocity * self.dt)

        # Simple contact and a conservative ground bounce clamp.
        grounded = self.position[:, 1] < 0.16
        self.position[:, 1].clamp_(min=0.16)
        self.velocity[:, 1] = torch.where(grounded & (self.velocity[:, 1] < 0),
                                         torch.zeros_like(self.velocity[:, 1]), self.velocity[:, 1])

        self.step_count += 1
        newly_took_off = (~self.has_taken_off) & (self.position[:, 1] >= 0.45)
        self.has_taken_off |= newly_took_off
        self.ground_wait = torch.where(self.has_taken_off, torch.zeros_like(self.ground_wait),
                                       self.ground_wait + 1)
        pos_error = self.position - self.target
        altitude_score = torch.exp(-pos_error[:, 1].abs() / 0.75)
        horizontal_error = torch.linalg.vector_norm(pos_error[:, [0, 2]], dim=-1)
        horizontal_score = torch.exp(-horizontal_error / 1.0)
        reward = 2.0 * (altitude_score - 0.5) + 0.5 * (horizontal_score - 0.5)
        reward -= 0.12 * self.velocity[:, 1].square().clamp(max=16)
        reward -= 0.04 * self.velocity[:, [0, 2]].square().sum(-1).clamp(max=25)
        reward -= 0.08 * self.attitude.square().sum(-1).clamp(max=9)
        reward -= 0.03 * self.angular_velocity.square().sum(-1).clamp(max=25)
        reward -= 0.01 * a[:, :3].square().sum(-1)
        reward -= 0.05 * (throttle - self.hover_throttle).square()
        reward -= 0.01 * self.ground_wait.clamp(max=50) * (~self.has_taken_off)
        crashed = (self.has_taken_off & (self.position[:, 1] <= 0.22)) | (self.position[:, 1] > 12.0) | (torch.linalg.vector_norm(self.position[:, [0, 2]], dim=-1) > 12.0)
        reward -= crashed.float() * 100.0
        timeout = self.step_count >= self.max_steps
        done = crashed | timeout
        obs = self.observation()
        # Auto-reset only completed environments; preserve terminal observations
        # separately for correct value bootstrapping in a production implementation.
        if done.any():
            self.reset(done)
        return obs, reward, done, {"crashed": crashed}
