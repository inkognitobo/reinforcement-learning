import inspect
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl.nn.utils import layer_init
from rl.utils import DEVICE
from rl.utils.types import Device


@dataclass
class Cfg:
    # === Rng ===
    seed: int = 1
    torch_deterministic: bool = True

    # Misc
    save_agent: bool = False
    capture_video: bool = False

    # === Run ===
    # Name of the algorithm inferred from file name
    algo_name: str = None
    # Name of the environment
    env_id: str = "CartPole-v1"
    # Name of the run, inferred from algo name, environment name, seed, and time
    run_name: str = None
    # Normalize observations centered at running mean with unit variance
    norm_obs: bool = False
    # If set, clip observations to `[-clip_obs, +clip_obs]`, should be used with normalization
    clip_obs: float = None
    # Normalize rewarsds s.t. their exponential running average has approximately fixed variance
    norm_rew: bool = False
    # If set, clip rewards to `[-clip_rew, +clip_rew]`, should be used with normalization
    clip_rew: float = None

    # === Algo ===
    # Number of environments to run in parallel to collect rollouts
    num_envs: int = 4
    # Total number of training time steps, distributed over available environments and batches
    total_timesteps: int = 500_000
    # Number of time steps in a single batch, 1 rollout ≙ 1 environment
    batch_size: int = 128
    # Number of minibatches that make up one batch
    num_minibatches: int = 4
    # Number of training epochs / updates on a single rollout (minibatches are shuffled each epoch)
    num_epochs: int = 4
    # Number of time steps in a single rollout, rollout comprises 1 batch per environment
    rollout_size = None
    # Number of time steps in a single minibatch, given by rollout size and number of minibatches
    minibatch_size = None
    # Number of rollouts in the training phase, given by total number of timesteps and rollout size
    num_iterations = None

    # Rate at which weights, etc. are updated
    learning_rate: float = 2.5e-4
    # If true, the learning rate converges to zero during training phase
    anneal_lr: bool = True
    # Discount parameter 1 for generalized advantage estimation
    gae_gamma: float = 0.99
    # Discount parameter 2 for generalized advantage estimation
    gae_lambda: float = 0.95
    # If true, the advantage is normalized
    norm_adv: bool = True
    # Clipping parameter for PPO objective
    clip_epsilon: float = 0.2
    # If true, the values are clipped before being used for value loss computation
    clip_values: bool = True
    # Coefficient to weight contribution of entropy to overall loss
    ent_coef: float = 0.01
    # Coefficient to weight contribution of value loss to overall loss
    vloss_coef: float = 0.5
    # Parameter for gradient clipping (Andrychowicz et al. (2021))
    max_grad_norm: float = 0.5
    # If set, approximated KL divergence is used as a stopping mechanism
    target_kl: float = None

    def update() -> None:
        Cfg.algo_name = os.path.basename(__file__)[:-len(".py")]
        Cfg.run_name = f"{time.strftime('%Y%m%d%H%M%S')}__{Cfg.algo_name}__{Cfg.env_id}__{Cfg.seed}"
        Cfg.rollout_size = int(Cfg.num_envs * Cfg.batch_size)
        Cfg.minibatch_size = int(Cfg.rollout_size // Cfg.num_minibatches)
        Cfg.num_iterations = Cfg.total_timesteps // Cfg.rollout_size

    def to_dict() -> dict[str, Any]:
        return {k: v
                for k, v in vars(Cfg).items() if not k.startswith("__")
                and not inspect.isfunction(v)}


def make_env(
        env_id: str,
        idx: int,
        norm_obs: bool = Cfg.norm_obs,
        clip_obs: float | None = Cfg.clip_obs,
        norm_rew: bool = Cfg.norm_rew,
        clip_rew: float | None = Cfg.clip_rew,
        capture_video: bool = Cfg.capture_video,
        run_name: str = Cfg.run_name,
):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(id=env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env=env, video_folder=os.path.join("videos", run_name))
        else:
            env = gym.make(id=env_id)

        if norm_obs:
            env = gym.wrappers.NormalizeObservation(env=env)
        if clip_obs is not None:
            env = gym.wrappers.TransformObservation(
                env=env,
                func=lambda obs: np.clip(obs, -clip_obs, clip_obs),
            )
        if norm_rew:
            env = gym.wrappers.NormalizeReward(env=env, gamma=Cfg.gae_gamma)
        if clip_rew is not None:
            env = gym.wrappers.TransformReward(
                env=env,
                func=lambda rew: np.clip(rew, -clip_rew, clip_rew),
            )

        env = gym.wrappers.RecordEpisodeStatistics(env=env)
        return env
    return thunk


class Agent(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: Device = DEVICE,
    ):
        super(Agent, self).__init__()

        self.critic = torch.nn.Sequential(
            layer_init(
                layer=torch.nn.Linear(in_features=in_features, out_features=64, device=device),
            ),
            torch.nn.Tanh(),
            layer_init(
                layer=torch.nn.Linear(in_features=64, out_features=64, device=device),
            ),
            torch.nn.Tanh(),
            layer_init(
                layer=torch.nn.Linear(in_features=64, out_features=1, device=device),
                weight_gain=1.0,
            ),
        )
        self.actor = torch.nn.Sequential(
            layer_init(
                layer=torch.nn.Linear(in_features=in_features, out_features=64, device=device),
            ),
            torch.nn.Tanh(),
            layer_init(
                layer=torch.nn.Linear(in_features=64, out_features=64, device=device)
            ),
            torch.nn.Tanh(),
            layer_init(
                layer=torch.nn.Linear(in_features=64, out_features=out_features, device=device),
                weight_gain=0.01),
        )

        # Register tensors as buffers
        # Available via `self.name` syntax, and saved alongside module's `state_dict`
        self.register_buffer(name="in_features", tensor=torch.tensor(in_features))
        self.register_buffer(name="out_features", tensor=torch.tensor(out_features))

        self.device = device
        self.to(device=self.device)

    def action(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.actor(tensor)

    def value(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.critic(tensor)

    def predict(
            self,
            tensor: torch.Tensor,
            action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(tensor)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(tensor)

        return action, logprob, entropy, value

    def save(self, f: str) -> None:
        torch.save(obj=self.state_dict(), f=f)

    @staticmethod
    def load(f: str) -> "Agent":
        statedict = torch.load(f=f)
        agent = Agent(in_features=statedict["in_features"].item(),
                      out_features=statedict["out_features"].item())
        return agent


class GeneralizedAdvantageEstimation(torch.nn.Module):
    def __init__(
            self,
            gae_gamma: float = Cfg.gae_gamma,
            gae_lambda: float = Cfg.gae_lambda,
            batch_size: int = Cfg.batch_size,
            device: Device = DEVICE,
    ):
        super(GeneralizedAdvantageEstimation, self).__init__()

        # Register tensors as buffers
        # Available via `self.name` syntax, and saved alongside module's `state_dict`
        self.register_buffer(name="gae_gamma", tensor=torch.tensor(gae_gamma))
        self.register_buffer(name="gae_lambda", tensor=torch.tensor(gae_lambda))
        self.register_buffer(name="batch_size", tensor=torch.tensor(batch_size))

        self.device = device
        self.to(device=self.device)

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        next_dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (rewards.shape == values.shape == next_values.shape == next_dones.shape)

        deltas = torch.zeros_like(rewards).to(device=self.device)
        advantages = torch.zeros_like(rewards).to(device=self.device)
        gae_discount = self.gae_gamma * self.gae_lambda
        running_gae = 0.0

        for t in reversed(range(self.batch_size)):
            # Compute Temporal Difference error (eq. 12 of PPO paper):
            # $\delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)$
            deltas[t] = rewards[t] + self.gae_gamma * next_values[t] * (~next_dones[t]) - values[t]

            # Compute Generalized Advantage Estimate (eq. 11 of PPO paper):
            # $\sum_{l=0}^{\infty} (\gamma * \lambda)^l \delta_{t+l}$
            advantages[t] = running_gae = deltas[t] + gae_discount * (~next_dones[t]) * running_gae

        # Estimate returns
        # $V_{targ} = est_returns = advantages + values
        returns = advantages + values

        return advantages, returns


class ClippedPPOLoss(torch.nn.Module):
    def __init__(
            self,
            clip_epsilon: float = Cfg.clip_epsilon,
            clip_values: bool = Cfg.clip_values,
            vloss_coef: float = Cfg.vloss_coef,
            ent_coef: float = Cfg.ent_coef,
            device: Device = DEVICE,
    ):
        super(ClippedPPOLoss, self).__init__()

        # Register tensors as buffers
        # Available via `self.name` syntax, and saved alongside module's `state_dict`
        self.register_buffer(name="clip_epsilon", tensor=torch.tensor(clip_epsilon))
        self.register_buffer(name="clip_values", tensor=torch.tensor(clip_values))

        self.register_buffer(name="vloss_coef", tensor=torch.tensor(vloss_coef))
        self.register_buffer(name="ent_coef", tensor=torch.tensor(ent_coef))

        self.device = device
        self.to(device=self.device)

    def forward(
        self,
        logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        values: torch.Tensor,
        new_values: torch.Tensor,
        new_entropies: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Compute ratio $r_t(\pi)$ (eq. 6 in PPO paper)
        # Same as exp(new) / exp(old)
        logratio = new_logprobs - logprobs
        ratio = logratio.exp()

        # L^CLIP: PPO objective (eq. 7 in PPO paper)
        # Paper defines the objective function; loss is the negative objective
        pg1 = advantages * ratio
        pg2 = advantages * torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        pg = torch.min(pg1, pg2).mean()

        # Value loss (eq. 9 in PPO paper)
        # Multiplying by 0.5 has the effect that when computing the derivative mean squared error
        # during the backward pass, the power of 2 and 0.5 will cancel out.
        if self.clip_values:
            v_loss1 = torch.nn.MSELoss(reduction="none")(new_values, returns)
            v_clipped = values + torch.clamp(
                new_values - values,
                -self.clip_epsilon,
                self.clip_epsilon,
            )
            v_loss2 = torch.nn.MSELoss(reduction="none")(v_clipped, returns)
            v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
        else:
            v_loss = 0.5 * torch.nn.MSELoss(reduction="none")(new_values, returns).mean()

        # Entropy bonus (eq. 9 in PPO paper)
        entropy_bonus = new_entropies.mean()

        # Total loss (to be minimised)
        objective = pg - self.vloss_coef * v_loss + self.ent_coef * entropy_bonus
        loss = -objective

        # Track additional statistics
        with torch.no_grad():
            # Calculate approximate KL divergence according to old approximator
            old_approx_kl = (-logratio).mean()
            # Calculate approximate KL divergence according to new approximator
            # (http://joschu.net/blog/kl-approx.html)
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
        stats = {
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clipfracs": clipfracs,
            "ratio": ratio,
            "policy_loss": -pg,
            "value_loss": v_loss,
            "entropy_bonus": entropy_bonus,
            "loss": loss,
            "objective": objective,
        }

        return loss, stats


def main():
    # Tracking
    writer = SummaryWriter(os.path.join("runs", "ppo", Cfg.env_id, Cfg.run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key,
                                      value in Cfg.to_dict().items()])),
    )

    # Seeding
    random.seed(Cfg.seed)
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)
    torch.backends.cudnn.deterministic = Cfg.torch_deterministic

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        env_fns=[make_env(Cfg.env_id, i)
                 for i in range(Cfg.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space
    in_features = np.array(obs_space.shape).prod()
    out_features = action_space.n
    assert isinstance(action_space, gym.spaces.Discrete), "only discrete action space supported"

    # Modules: policy, optimizer, gae, loss
    agent = Agent(in_features=in_features, out_features=out_features)
    optimizer = torch.optim.Adam(params=agent.parameters(), lr=Cfg.learning_rate, eps=1e-5)
    generalized_advantage_estimation = GeneralizedAdvantageEstimation()
    clipped_ppo_loss = ClippedPPOLoss()

    # Rollout storage
    # Using shape `[BATCH_SIZE, NUM_ENVS, ...,]`, allows to index by step
    observations = torch.zeros((Cfg.batch_size, Cfg.num_envs, *obs_space.shape)).to(device=DEVICE)
    actions = torch.zeros((Cfg.batch_size, Cfg.num_envs, *action_space.shape)).to(device=DEVICE)
    logprobs = torch.zeros((Cfg.batch_size, Cfg.num_envs)).to(device=DEVICE)
    rewards = torch.zeros((Cfg.batch_size, Cfg.num_envs)).to(device=DEVICE)
    dones = torch.zeros((Cfg.batch_size, Cfg.num_envs)).to(dtype=torch.bool, device=DEVICE)
    values = torch.zeros((Cfg.batch_size, Cfg.num_envs)).to(device=DEVICE)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=Cfg.seed)
    next_obs = torch.Tensor(next_obs).to(device=DEVICE)
    next_done = torch.zeros(Cfg.num_envs).to(dtype=torch.bool, device=DEVICE)

    agent.train()
    pbar = tqdm(range(1, Cfg.num_iterations + 1))
    for iteration in pbar:
        if Cfg.anneal_lr:
            # Annealing the rate if instructed to do so
            frac = 1.0 - (iteration - 1.0) / Cfg.num_iterations
            lrnow = frac * Cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout (fixed-length trajectory segment)
        # Rollout := 1 batch per environment
        # Resetting the environment is taken care of by `VectorEnv`
        for step in range(0, Cfg.batch_size):
            global_step += Cfg.num_envs
            observations[step] = next_obs
            dones[step] = next_done

            # Predict action
            # Each tensor contains one per environment
            with torch.no_grad():
                action, logprob, new_entropies, value = agent.predict(tensor=next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.view(-1)

            # Step the environment
            # Each tensor contains one per environment
            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            next_done = np.logical_or(term, trunc)
            rewards[step] = torch.tensor(reward).to(device=DEVICE).view(-1)
            next_obs = torch.Tensor(next_obs).to(device=DEVICE)
            next_done = torch.Tensor(next_done).to(dtype=torch.bool, device=DEVICE)

            # Tracking
            # NOTE: `VecEnv` specific
            if len(info.items()) > 0:
                for i in range(Cfg.num_envs):
                    if info["_episode"][i] and "episode" in info:
                        # `info["_episode"]` records for each environment if the episode ended,
                        # `info["episode"]` has a dictionary, where each entry contains episode
                        # stats for each of the environment; only those finished have actual values
                        episodic_return = info["episode"]["r"][i]
                        episodic_length = info["episode"]["l"][i]

                        writer.add_scalar(
                            tag="charts/episodic_return",
                            scalar_value=episodic_return,
                            global_step=global_step,
                        )
                        writer.add_scalar(
                            tag="charts/episodic_length",
                            scalar_value=episodic_length,
                            global_step=global_step,
                        )
                        pbar.set_description(
                            desc=f"{global_step}: "
                            f"last episodic return: {episodic_return:.2f}, "
                            f"current learning rate: {optimizer.param_groups[0]['lr']:.5f}"
                        )

        with torch.no_grad():
            # Bootstrap final next value using value of next observation
            # (last observation returned by environment, not part of rollout)
            next_dones = torch.cat(tensors=(dones[1:], next_done.unsqueeze(dim=0)))
            next_value = agent.value(next_obs).reshape(1, -1)
            next_values = torch.cat(tensors=(values[1:], next_value))

            # Generalized Advantage Estimation
            advantages, returns = generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                next_values=next_values,
                next_dones=next_dones,
            )

        # Flatten the batch
        # From here on out the behaviour is equivalent to the single environment case
        # NOTE: Only requried for vectorized environments, I think
        batch_observations = observations.reshape((-1, *obs_space.shape))
        batch_actions = actions.reshape((-1, *action_space.shape))
        batch_logprobs = logprobs.reshape(-1)
        batch_values = values.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)

        # Optimizing step
        clipfracs = torch.zeros((Cfg.num_epochs,))
        for epoch in range(Cfg.num_epochs):
            # Shuffle batch indices
            batch_indices = torch.randperm(Cfg.rollout_size)
            for start in range(0, Cfg.rollout_size, Cfg.minibatch_size):
                # Select a minibatch
                end = start + Cfg.minibatch_size
                minibatch_indices = batch_indices[start:end]
                minibatch_observations = batch_observations[minibatch_indices]
                minibatch_actions = batch_actions[minibatch_indices]
                minibatch_logprobs = batch_logprobs[minibatch_indices]
                minibatch_values = batch_values[minibatch_indices]
                minibatch_advantages = batch_advantages[minibatch_indices]
                minibatch_returns = batch_returns[minibatch_indices]

                # Normalize advantages
                # Add 1e-8 for numerical stability (avoid risk of divide-by-zero)
                if Cfg.norm_adv:
                    mean = minibatch_advantages.mean()
                    std = minibatch_advantages.std()
                    minibatch_advantages = (minibatch_advantages - mean) / (std + 1e-8)

                _, new_logprobs, new_entropies, new_values = agent.predict(
                    minibatch_observations,
                    minibatch_actions.long()
                )
                new_values = new_values.view(-1)

                loss, stats = clipped_ppo_loss(
                    logprobs=minibatch_logprobs,
                    new_logprobs=new_logprobs,
                    values=minibatch_values,
                    new_values=new_values,
                    new_entropies=new_entropies,
                    advantages=minibatch_advantages,
                    returns=minibatch_returns,
                )

                # Actual optimizer step
                # Clip gradient for small performance boost (Andrychowicz et al. (2021))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=agent.parameters(), max_norm=Cfg.max_grad_norm)
                optimizer.step()

            # Track additional statistics
            clipfracs[epoch] = stats["clipfracs"]

            # Optionally when target KL divergence is reached,
            # stop all epoch updateson this rollout and start collecting the next
            if Cfg.target_kl is not None and stats["approx_kl"] > Cfg.target_kl:
                break

        # Explained variance:
        # (https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score)
        # Indicator if the value function is a good predictor of the returns
        # `explained_var = 1` => perfect predictions
        # `explained_var = 0` => imperfect predictions
        # Target output: Empirical returns, estimated using the advantages and values
        # Predicted output: Predicted values, computed by the critic network
        output = batch_values
        target_output = batch_returns
        explained_var = 1 - ((target_output - output).var() / (target_output.var() + 1e-8))

        # Tracking
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_estimate", new_values.mean(), global_step)
        writer.add_scalar("losses/policy_loss", stats["policy_loss"].item(), global_step)
        writer.add_scalar("losses/value_loss", stats["value_loss"].item(), global_step)
        writer.add_scalar("losses/entropy", stats["entropy_bonus"].item(), global_step)
        writer.add_scalar("losses/mean_advantage", batch_advantages.mean(), global_step)
        writer.add_scalar("losses/old_approx_kl", stats["old_approx_kl"].item(), global_step)
        writer.add_scalar("losses/approx_kl", stats["approx_kl"].item(), global_step)
        writer.add_scalar("losses/clipfrac", clipfracs.mean(), global_step)
        writer.add_scalar("losses/ratio", stats["ratio"].mean(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # Save model
    if Cfg.save_agent:
        agent.save(f=os.path.join("models", Cfg.run_name, "agent.pt"))

    envs.close()
    writer.close()


if __name__ == "__main__":
    # Compute derived values for configuration
    Cfg.update()

    for seed in [1, 12, 123, 1234]:
        Cfg.seed = seed
        Cfg.update()
        main()

    exit()
