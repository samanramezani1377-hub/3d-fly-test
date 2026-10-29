"""CUDA-first PPO runner for the experimental batched GPU simulator.

Requires an NVIDIA CUDA GPU. It refuses CPU fallback. This is a separate
experimental path and does not modify the existing DroneCore/SB3 runner.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from cuda_hover_batch import CudaHoverBatch


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=40_000_000)
    p.add_argument("--n-envs", type=int, default=40)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-freq", type=int, default=100_000)
    p.add_argument("--output-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--run-name", default="cuda-ppo-hover")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    return p.parse_args()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=12, action_dim=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.policy_mean = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, obs):
        features = self.trunk(obs)
        return self.policy_mean(features), self.value(features).squeeze(-1)

    def act(self, obs):
        mean, value = self(obs)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        raw_action = dist.sample()
        action = raw_action.tanh()
        # Change-of-variables correction for tanh-squashed Gaussian.
        log_prob = (dist.log_prob(raw_action) - torch.log(1 - action.square() + 1e-6)).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        mean, value = self(obs)
        actions = actions.clamp(-0.999999, 0.999999)
        raw = torch.atanh(actions)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        log_prob = (dist.log_prob(raw) - torch.log(1 - actions.square() + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


def main():
    args = args_parser()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required. No CPU fallback will be used.")
    if min(args.timesteps, args.n_envs, args.n_steps, args.batch_size, args.epochs, args.save_freq) <= 0:
        raise SystemExit("All counts and frequencies must be positive.")
    rollout_size = args.n_envs * args.n_steps
    if args.batch_size > rollout_size or rollout_size % args.batch_size:
        raise SystemExit("n-envs * n-steps must be evenly divisible by batch-size.")
    if args.timesteps % args.n_envs:
        print("Note: final rollout may overshoot requested transitions to finish a vector step.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = CudaHoverBatch(args.n_envs, device=str(device))
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf = torch.empty((args.n_steps, args.n_envs, 12), device=device)
    act_buf = torch.empty((args.n_steps, args.n_envs, 4), device=device)
    logp_buf = torch.empty((args.n_steps, args.n_envs), device=device)
    rew_buf = torch.empty((args.n_steps, args.n_envs), device=device)
    done_buf = torch.empty((args.n_steps, args.n_envs), device=device)
    val_buf = torch.empty((args.n_steps, args.n_envs), device=device)

    obs = env.observation()
    global_step = 0
    update = 0
    start = time.perf_counter()
    next_save = args.save_freq
    metadata = {
        "algorithm": "PPO",
        "backend": "torch-cuda-batched-surrogate",
        "device_required": "cuda",
        "shared_policy": True,
        "parallel_environments": args.n_envs,
        "n_steps_per_environment": args.n_steps,
        "batch_size": args.batch_size,
        "target_environment_transitions": args.timesteps,
        "rollout_size": rollout_size,
        "epochs": args.epochs,
        "physics_parity": False,
        "warning": "GPU-native surrogate, not equivalent to pinned DroneCore; do not treat as validated flight physics.",
        "python_cpu_role": "setup, filesystem, periodic scalar logging/checkpoint only; simulation, rewards, rollout tensors and PPO updates are CUDA tensors",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        while global_step < args.timesteps:
            model.eval()
            for t in range(args.n_steps):
                obs_buf[t].copy_(obs)
                with torch.no_grad():
                    action, logp, value = model.act(obs)
                act_buf[t].copy_(action)
                logp_buf[t].copy_(logp)
                val_buf[t].copy_(value)
                next_obs, reward, done, _ = env.step(action)
                rew_buf[t].copy_(reward)
                done_buf[t].copy_(done.float())
                obs = next_obs
                global_step += args.n_envs

            with torch.no_grad():
                _, next_value = model(obs)
                advantages = torch.zeros_like(rew_buf)
                last_gae = torch.zeros(args.n_envs, device=device)
                for t in reversed(range(args.n_steps)):
                    next_nonterminal = 1.0 - done_buf[t]
                    if t == args.n_steps - 1:
                        next_values = next_value
                    else:
                        next_values = val_buf[t + 1]
                    delta = rew_buf[t] + args.gamma * next_values * next_nonterminal - val_buf[t]
                    last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + val_buf

            b_obs = obs_buf.reshape(-1, 12)
            b_actions = act_buf.reshape(-1, 4)
            b_logp = logp_buf.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)
            model.train()
            losses = []
            for _ in range(args.epochs):
                permutation = torch.randperm(rollout_size, device=device)
                for start_idx in range(0, rollout_size, args.batch_size):
                    idx = permutation[start_idx:start_idx + args.batch_size]
                    new_logp, entropy, new_value = model.evaluate_actions(b_obs[idx], b_actions[idx])
                    log_ratio = new_logp - b_logp[idx]
                    ratio = log_ratio.exp()
                    pg1 = -b_adv[idx] * ratio
                    pg2 = -b_adv[idx] * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                    policy_loss = torch.maximum(pg1, pg2).mean()
                    value_loss = 0.5 * (new_value - b_returns[idx]).square().mean()
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + 0.5 * value_loss + 0.005 * entropy_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    losses.append(loss.detach())

            update += 1
            if global_step >= next_save or global_step >= args.timesteps:
                # CPU transfer occurs only at checkpoint/logging boundaries.
                checkpoint = {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "update": update,
                    "config": vars(args) | {"output_dir": str(args.output_dir)},
                }
                path = run_dir / f"checkpoint_{global_step}.pt"
                torch.save(checkpoint, path)
                mean_reward = rew_buf.mean().item()
                mean_loss = torch.stack(losses).mean().item() if losses else float("nan")
                elapsed = time.perf_counter() - start
                print(f"step={global_step:,} update={update} mean_reward={mean_reward:.4f} loss={mean_loss:.4f} env_steps_per_sec={global_step/elapsed:.1f} checkpoint={path}", flush=True)
                next_save += args.save_freq

        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / "final_model_state_dict.pt")
        print(f"Finished {global_step:,} transitions. Final model: {run_dir / 'final_model_state_dict.pt'}")
    finally:
        del env, model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
