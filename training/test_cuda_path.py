"""CUDA smoke test for the experimental batched simulator and PPO model."""
import torch
from cuda_hover_batch import CudaHoverBatch
from train_ppo_cuda import ActorCritic

def main():
    assert torch.cuda.is_available(), "Run this test on a CUDA-enabled runner/Colab GPU."
    device = "cuda"
    env = CudaHoverBatch(n_envs=40, device=device)
    obs = env.observation()
    assert obs.device.type == "cuda" and obs.shape == (40, 12)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(128):
        action, log_prob, value = model.act(obs)
        obs, reward, done, _ = env.step(action)
        assert obs.device.type == reward.device.type == done.device.type == "cuda"
        assert torch.isfinite(obs).all()
        assert torch.isfinite(reward).all()
        loss = -(log_prob.mean()) + 0.5 * value.square().mean() - 0.001 * reward.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    print("CUDA smoke test passed: 40 environments, physics/reward/rollouts/model update on CUDA.")
    print("NOTE: this verifies device placement and finite values, not parity with DroneCore.")

if __name__ == "__main__":
    main()
