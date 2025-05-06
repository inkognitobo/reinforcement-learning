import os

import torch
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np

from typing import Any

from tqdm import tqdm

from src.nn.utils import layer_init
from src.utils import DEVICE
from src.utils.types import Device


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
                torch.nn.Linear(in_features=in_features, out_features=64, device=device),
            ),
            torch.nn.Tanh(),
            layer_init(
                torch.nn.Linear(in_features=64, out_features=64, device=device),
            ),
            torch.nn.Tanh(),
            layer_init(
                torch.nn.Linear(in_features=64, out_features=1, device=device),
                std=1.0,
            ),
        )

        self.actor = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(in_features=in_features, out_features=64, device=device),
            ),
            torch.nn.Tanh(),
            layer_init(
                torch.nn.Linear(in_features=64, out_features=64, device=device),
            ),
            torch.nn.Tanh(),
            layer_init(
                torch.nn.Linear(in_features=64, out_features=out_features, device=device),
                std=0.01,
            ),
        )

        # Register tensors as buffers
        # Available via `self.name` syntax, and saved alongside module's `state_dict`
        self.register_buffer(name="in_features", tensor=torch.tensor(in_features))
        self.register_buffer(name="out_features", tensor=torch.tensor(out_features))

        self.device = device
        self.to(device=self.device)

    def predict(
            self,
            tensor: torch.Tensor,
            action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = tensor.to(self.device)
        action = action.to(self.device) if action is not None else None

        logits = self.actor(tensor)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(value=action)
        entropy = probs.entropy()
        value = self.critic(tensor)

        return action, logprob, entropy, value

    def action(self, tensor: torch.Tensor):
        action, logprob, entropy, value = self.predict(tensor=tensor)
        return action

    def value(self, tensor: torch.Tensor):
        action, logprob, entropy, value = self.predict(tensor=tensor)
        return value

    @staticmethod
    def load(f: str) -> "Agent":
        statedict = torch.load(f=f)
        agent = Agent(in_features=statedict["in_features"].item(),
                      out_features=statedict["out_features"].item())
        return agent


class GeneralizedAdvantageEstimation(torch.nn.Module):
    def __init__(
            self,
            gae_gamma: float = 0.99,
            gae_lambda: float = 0.95,
            device: Device = DEVICE,
    ):
        super().__init__()

        # Register tensors as buffers
        self.register_buffer(name="gae_gamma", tensor=torch.tensor(gae_gamma))
        self.register_buffer(name="gae_lambda", tensor=torch.tensor(gae_lambda))

        self.device = device
        self.to(device=self.device)

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        next_dones: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (rewards.shape == values.shape == next_values.shape == next_dones.shape)

        with torch.no_grad():
            next_not_dones = 1.0 - next_dones

            advantages = torch.zeros_like(rewards).to(device=self.device)
            deltas = torch.zeros_like(rewards).to(device=self.device)
            gae_discount = self.gae_gamma * self.gae_lambda
            running_gae = 0.0

            for t in reversed(range(batch_size)):
                # Compute Temporal Difference error (eq. 12 of PPO paper):
                # $\delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)$
                delta = rewards[t] + self.gae_gamma * next_values[t] * next_not_dones[t] - values[t]
                deltas[t] = delta

                # Compute Generalized Advantage Estimate (eq. 11 of PPO paper):
                # $\sum_{l=0}^{\infty} (\gamma * \lambda)^l \delta_{t+l}$
                advantages[t] = running_gae = delta + gae_discount * next_not_dones[t] * running_gae

            # Estimate returns
            returns = advantages + values

        return advantages, returns


class ClipPPOLoss(torch.nn.Module):
    def __init__(
            self,
            clip_epsilon: float = 0.2,
            clip_value: bool = False,
            value_coef: float = 0.5,
            entropy_coef: float = 0.01,
            norm_adv: bool = True,
            device: Device = DEVICE,
    ):
        super().__init__()

        # Register tensors as buffers
        self.register_buffer(name="clip_epsilon", tensor=torch.tensor(clip_epsilon))
        self.register_buffer(name="clip_value", tensor=torch.tensor(clip_value))

        self.register_buffer(name="value_coef", tensor=torch.tensor(value_coef))
        self.register_buffer(name="entropy_coef", tensor=torch.tensor(entropy_coef))

        self.register_buffer(name="norm_adv", tensor=torch.tensor(norm_adv))

        self.device = device
        self.to(device=self.device)

    def forward(
        self,
        # These are tensors containing data of a single minibatch
        logprobs: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        new_logprobs: torch.Tensor,
        new_entropies: torch.Tensor,
        new_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Compute ratio $r_t(\pi)$ (eq. 6 in PPO paper)
        # Last step is the same as exp(new) / exp(old)
        logratio = new_logprobs - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            fraction_clipped = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

        # Normalize advantages
        # Add 1e-8 for numerical stability (avoid risk of divide-by-zero)
        if self.norm_adv:
            mean = advantages.mean()
            std = advantages.std()
            advantages = (advantages - mean) / (std + 1e-8)

        # Policy loss (eq. 7 in PPO paper)
        # Paper defines the objective function; loss is the negative objective,
        # hence we also take the maximum instead of the minimum.
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss (eq. 9 in PPO paper)
        # Multiplying by 0.5 has the effect that when computing the derivative mean squared error (MSE)
        # during the backward pass, the power of 2 and 0.5 will cancel out.
        if self.clip_value:
            v_loss_unclipped = torch.nn.MSELoss(reduction="none")(values, returns)
            v_clipped = values + torch.clamp(
                new_values - values,
                -self.clip_epsilon,
                self.clip_epsilon,
            )
            v_loss_clipped = torch.nn.MSELoss(reduction="none")(v_clipped, returns)
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((values - returns) ** 2).mean()

        # Entropy bonus (eq. 9 in PPO paper)
        entropy_bonus = new_entropies.mean()

        # Total loss
        loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy_bonus

        # Compute debug stats
        # Approximate Kullback-Leibler divergence
        # Fraction of the training data that triggered the clipped objective
        stats = {
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "fraction_clipped": fraction_clipped,
        }

        return loss, stats


def evaluate(
        agent: Agent,
        env_id: str,
        num_episodes: int = 10,
):
    training = agent.training
    agent.eval()

    env = gym.make(env_id)

    returns = []
    lengths = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        obs = torch.tensor(data=obs).to(device=DEVICE)
        done = torch.zeros((1,)).to(device=DEVICE)
        ep_return = 0
        ep_length = 0
        while True:
            with torch.no_grad():
                action = agent.action(tensor=obs)
            obs, reward, terminated, truncated, info = env.step(
                action=action.cpu().numpy(),
            )

            done = torch.tensor(data=terminated or truncated).to(device=DEVICE)
            obs = torch.tensor(data=obs).to(device=DEVICE)

            ep_return += reward
            ep_length += 1

            if done:
                break
        returns += [ep_return]
        lengths += [ep_length]

    if training:
        agent.train()

    stats = {
        "avg_return": torch.tensor(data=returns).float().mean(),
        "avg_length": torch.tensor(data=lengths).float().mean(),
    }

    return stats


if __name__ == "__main__":
    import gymnasium as gym

    # NOTE: Naming is as if this was single environment:
    # - single step -> singular in naming ("obs", "action", "reward"), even though
    # these may be vectors containing entries for each environment
    # - batch -> plural in naming ("observations", "actions", "rewards")

    EXPERIMENT_NAME = "ppo/0002"
    os.makedirs(f"models/{EXPERIMENT_NAME}", exist_ok=True)
    os.makedirs(f"runs/{EXPERIMENT_NAME}", exist_ok=True)

    NUM_ENVS = 5
    TOTAL_TIMESTEPS = 100_000
    LEARNING_RATE = 2.5e-4
    BATCH_SIZE = 128
    NUM_BATCHES = int(TOTAL_TIMESTEPS // BATCH_SIZE)
    NUM_MINIBATCHES = 4
    MINIBATCH_SIZE = int(BATCH_SIZE // NUM_MINIBATCHES)
    ANNEAL_LR = True
    NUM_EPOCHS = 4
    MAX_GRAD_NORM = 0.5

    SEED = 24
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    # Write statistics
    writer = SummaryWriter(f"runs/{EXPERIMENT_NAME}")

    # Environment
    env_id = "CartPole-v1"
    envs = gym.vector.SyncVectorEnv(
        env_fns=[lambda:
                 gym.wrappers.RecordEpisodeStatistics(gym.make(id=env_id))
                 for i in range(NUM_ENVS)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action spaces ok"
    observation_space = envs.single_observation_space
    action_space = envs.single_action_space
    in_features = observation_space.shape[-1]
    out_features = action_space.n

    # Agent
    agent = Agent(in_features=in_features, out_features=out_features)
    optimizer = torch.optim.Adam(
        params=agent.parameters(),
        lr=LEARNING_RATE,
        eps=1e-5,  # 1e-5 / 1e-8 / 1e-7
        maximize=False,
    )
    gae_fn = GeneralizedAdvantageEstimation()
    loss_fn = ClipPPOLoss()

    # Memory
    # Using shape (BATCH_SIZE, NUM_ENVS, ...,) allows indexing by step
    observations = torch.zeros((BATCH_SIZE, NUM_ENVS, *observation_space.shape)).to(device=DEVICE)
    actions = torch.zeros((BATCH_SIZE, NUM_ENVS, *action_space.shape)).to(device=DEVICE)
    logprobs = torch.zeros((BATCH_SIZE, NUM_ENVS,)).to(device=DEVICE)
    rewards = torch.zeros((BATCH_SIZE, NUM_ENVS,)).to(device=DEVICE)
    dones = torch.zeros((BATCH_SIZE, NUM_ENVS,)).to(device=DEVICE)
    values = torch.zeros((BATCH_SIZE, NUM_ENVS,)).to(device=DEVICE)

    # Other
    global_step = 0
    obs, _ = envs.reset(seed=SEED)
    obs = torch.tensor(data=obs).to(device=DEVICE)
    done = torch.zeros((1,)).to(device=DEVICE)

    agent.train()
    pbar = tqdm(range(1, NUM_BATCHES + 1))
    # Number of updates = number of batches
    for b in pbar:
        if ANNEAL_LR:
            # Anneal learning rate
            frac = 1.0 - (b - 1.0) / NUM_BATCHES
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout (fixed-length trajectory segment)
        # Resetting the environments is taken care off by `VectorEnv`
        for step in range(0, BATCH_SIZE):
            global_step += NUM_ENVS  # 1 step per environment
            observations[step] = obs
            dones[step] = done

            with torch.no_grad():
                action, logprob, entropy, value = agent.predict(tensor=obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.squeeze()

            obs, reward, terminated, truncated, info = envs.step(actions=action.cpu().numpy())
            done = torch.tensor(data=np.logical_or(terminated, truncated)).to(device=DEVICE)
            obs = torch.tensor(data=obs).to(device=DEVICE)
            rewards[step] = torch.tensor(data=reward).to(device=DEVICE)

            # print(info)
            # NOTE: `VecEnv` specific
            if len(info.items()) > 0:
                for i in range(NUM_ENVS):
                    if info["_episode"][i] and "episode" in info:
                        episodic_return = info["episode"]["r"][i]
                        episodic_length = info["episode"]["l"][i]
                        time = info["episode"]["t"][i]

                        writer.add_scalar(
                            tag="charts/episodic_return",
                            scalar_value=episodic_return, global_step=global_step,
                        )
                        writer.add_scalar(
                            tag="charts/episodic_length",
                            scalar_value=episodic_length, global_step=global_step,
                        )
                        pbar.set_description(
                            desc=f"{global_step}: "
                            f"last episodic return: {episodic_return:.2f}, "
                            f"current learning rate: {optimizer.param_groups[0]['lr']:.5f}"
                        )

        # Bootstrap next value if not done
        with torch.no_grad():
            next_value = agent.value(obs).reshape(1, -1)
            next_value = torch.where(done, next_value, torch.zeros_like(next_value))
        next_values = torch.cat(tensors=(values[1:], next_value))
        next_dones = torch.cat(tensors=(dones[1:], done.unsqueeze(dim=0)))

        # Compute advantage
        advantages, returns = gae_fn(
            rewards=rewards,
            values=values,
            next_values=next_values,
            next_dones=next_dones,
            batch_size=BATCH_SIZE,
        )

        # Flatten batch data
        # From here on out the behaviour is equivalent to the single environment case
        # NOTE: Only requried for vectorized environments, I think
        batch_observations = observations.reshape((-1, *observation_space.shape))
        batch_logprobs = logprobs.reshape(-1)
        batch_actions = actions.reshape((-1, *action_space.shape))
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        # Update model
        epoch_losses = torch.zeros((NUM_EPOCHS,)).to(device=DEVICE)
        for epoch in range(NUM_EPOCHS):
            # Shuffle batch indices
            batch_indices = torch.randperm(BATCH_SIZE)
            batch_losses = []
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                # Select a minibatch
                end = start + MINIBATCH_SIZE
                minibatch_indices = batch_indices[start:end]

                _, logprobs_prime, entropies_prime, values_prime = agent.predict(
                    batch_observations[minibatch_indices],
                    batch_actions.long()[minibatch_indices],
                )

                # Loss
                loss, stats = loss_fn(
                    logprobs=batch_logprobs[minibatch_indices],
                    values=batch_values[minibatch_indices],
                    advantages=batch_advantages[minibatch_indices],
                    returns=batch_returns[minibatch_indices],
                    new_logprobs=logprobs_prime,
                    new_entropies=entropies_prime,
                    new_values=values_prime,
                )
                batch_losses += [loss]

                optimizer.zero_grad()
                loss.backward()
                # Clip gradient for small performance boost (Andrychowicz et al. (2021))
                torch.nn.utils.clip_grad_norm_(
                    parameters=agent.parameters(),
                    max_norm=MAX_GRAD_NORM,
                )
                optimizer.step()

            epoch_losses[epoch] = torch.tensor(data=batch_losses).mean()

        avg_epoch_loss = epoch_losses.mean()
        writer.add_scalar(
            tag="charts/learning_rate",
            scalar_value=optimizer.param_groups[0]["lr"],
            global_step=global_step,
        )
        writer.add_scalar(
            tag="losses/total_loss",
            scalar_value=avg_epoch_loss,
            global_step=global_step,
        )
        writer.add_scalar(
            tag="losses/old_approx_kl",
            scalar_value=stats["old_approx_kl"],
            global_step=global_step,
        )
        writer.add_scalar(
            tag="losses/approx_kl",
            scalar_value=stats["approx_kl"],
            global_step=global_step,
        )
        writer.add_scalar(
            tag="losses/fraction_clipped",
            scalar_value=stats["fraction_clipped"],
            global_step=global_step,
        )

    # Final Evaluation
    stats = evaluate(agent=agent, env_id=env_id, num_episodes=100)
    print(
        "Final evaluation: \n"
        f"Avg episode return: {stats['avg_return']:.2f} \n"
        f"Avg episode length: {stats['avg_length']:.2f} \n"
    )

    # Save model
    torch.save(obj=agent.state_dict(), f=f"models/{EXPERIMENT_NAME}/agent.pt")

    envs.close()
    exit()
