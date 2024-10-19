# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        # when we say a distribution, we are referring to PDF(probability density function) of the distribution
        # pdf x-axis is the event(action here is action),
        # and the y-axis (y-axis meaning)does not care much here with one restriction =>
        # the area under the curve from the x-axis[0, x] < 1 and the area_all = 1
        # refer to https://www.youtube.com/watch?v=rnBbYsysPaU
        probs = Categorical(logits=logits)
        if action is None:
            # here it involves sth called Inverse Transform Sampling using CDF(cumulative density function)
            # and differentiation(seeking the derivative) and integration are opposite operation of each other
            # here we randomly pick a probability from current CDF (whose y-axis [0, 1] which is a uniform distribution)
            # and then calculate the action (x-axis value)
            # current CDF means with current weights and biases network
            action = probs.sample()
        # probs.log_prob(action)
        # using PDF, plugin in action(x) to get probablity(p)
        # and then return log(p) and using logarithm afterwards brings many benefits to ML,
        # like no gradient explosion (no more 0.1*0.001*0.001*... =>0.000...0001 computer could not store/represent that)
        # refer to https://www.youtube.com/watch?v=rnBbYsysPaU
        #
        # probs.entropy()
        # the Average surprise of states in a distribution
        # H = ∑( Ps * log(1/Ps) )
        # refer to https://www.youtube.com/watch?v=KHVR587oW8I&t=1009s
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        # it works like (128, 4) + (4,) => (128, 4, 4)
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # the number of iterations (computed in runtime)
    # args.num_iterations = args.total_timesteps // args.batch_size
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        # the number of steps to run in each environment per policy rollout
        # num_steps = 128
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # episode => agent take steps until the env ends( terminates or truncates)
                        # For Gymnasium, the “agent-environment-loop” is implemented below for a single episode (until the environment ends)
                        # refer to https://gymnasium.farama.org/introduction/basic_usage/
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # At GAE formula
                # refer to
                # https://medium.com/deepgamingai/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-22337981f815
                # https://github.com/opendilab/PPOxFamily/blob/main/chapter7_tricks/gae.py
                #
                # next_value *= (1 - done) # If done equals 1, it indicates the end of an episode, thus the next state value should be 0 (means `lastgaelam=0` again).
                # delta = reward + gamma * next_value - value # Calculate the temporal difference (TD) error for each time step.
                # factor = gamma * lambda_ * (1 - traj_flag) # # Set the GAE decay factor. If traj_flag equals 1, the factor will be 0. Otherwise, the factor is gamma * lambda.
                # gae[t] = delta[t] + gamma * lambda * gae[t+1]
                # gae_item = torch.zeros_like(value[0]) => [0,0,0,...]
                # for t in reversed(range(reward.shape[0])):
                #     gae_item[t] = delta[t] + factor[t] * gae_item
                #     adv[t] = gae_item
                #
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    # gae_lambda is a smoothing parameter (~0.95)
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            # The RL framework: An agent senses the current state (s) of its environment.
            # The agent takes an action (a), selected from the action set (A), using policy (π),
            # and receives an immediate reward (r),
            # with the goal of receiving large future return (R).
            # http://www.breloff.com/DeepRL-OnlineGAE/
            returns = advantages + values  # for a large future returns

        # flatten the batch
        # obs.shape => (128, 4, 4) # (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        # with obs.reshape((-1, 4)) => (128*4, 4) => (512, 4) it'd be like flattening the rollout data
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # epoch and full batch vs mini batch
        # refer to https://blog.csdn.net/qq_38343151/article/details/102886304
        #
        # TODO, would not the total_loss for the first time is already 0, small enough?
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # refer to https://www.youtube.com/watch?v=KHVR587oW8I
                # entropy, cross-entropy (CE) VS square entropy(SE)
                # Ps means probs from dataset, Qs means probs from model approximation
                # cross-entropy => H(P, Q)= sum( Ps * log(1/Qs))
                # square-entropy => avg( sum( (Ps-Qs)**2 ) )
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                # in ml, log(x) = ln(x), just using e-base
                # we want `ratio=new/old` = `e to ln(new/old)` = `e to (ln(new) - ln(old))`
                # and we are now `e to (ln(new) - ln(old))` => `e to ln(new/old)` => `ratio=new/old`
                logratio = (
                    newlogprob - b_logprobs[mb_inds]
                )  # first time all [0, 0, ...]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # refer to KL divergence
                    # https://www.youtube.com/watch?v=KHVR587oW8I
                    # https://www.youtube.com/watch?v=sjgZxuCm_8Q
                    # https://www.youtube.com/watch?v=q0AkK8aYbLY
                    # KL Divergence is to tell how similar two probability distributions are
                    # the smaller, the more similar
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # clipfracs is used to monitor how often the policy updates are being clipped
                    # By analyzing clipfracs, one can understand how often clipping occurs during training.
                    # Frequent clipping might indicate that the learning rate or update steps are too aggressive,
                    # whereas infrequent clipping may imply under-tuning.
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:  # default is True
                    # a common technique for normalizing
                    # (x - x_mean) / x_std effectively scaling the x to have a standard deviation of 1
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std()
                        + 1e-8  # just make sure that this is not divided by 0
                    )

                # Policy loss pg_loss -> Policy Gradient Loss
                # 1. the minus sign `-`.
                # In reinforcement learning, particularly in policy gradient methods, the objective is to maximize the expected return.
                # This is usually expressed as minimizing the negative of the expected return to fit the usual optimization framework.
                # If we denote the objective function as J(θ), our goal is to maximize this with respect to parameters θ.
                # However, optimization libraries commonly minimize functions, so it's standard practice to multiply by -1 (i.e., minimize −J(θ)).
                # the minus sign `-` flips the direction of the optimization
                # 2. min vs max
                # due to 1,
                # using `-mb_advantages` => `torch.max` or `mb_advantages` => `-.torch.min`
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                # assuming
                # `ratio = [1.4, 0.6, 0.9]`
                # `clamped_ratio = [1.2, 0.8, 0.9] with clip_coef =0.2`
                # after `torch.max(ratio, clamped_ratio) => [1.4, 0.8, 0.9]`
                # isn't 1.4 considered as moves too far from old policy?
                #
                # When the ratio is [1.4],
                # The update seems large and potentially destabilizing but bringing more exploration in the problem solution space.
                # While mathematically torch.max prioritizes the larger value,
                # the choice to use pg_loss as torch.max(pg_loss1, pg_loss2) is conceptual
                # It only avoids the negative(bad) direction bringing too much impact,
                # since `min(ratio*adv, clipped_ratio*adv)` in fact only prioritizes one direction(-∞, upper bound]
                # and In practice, unexpected large values are usually rare and handled by various training strategies
                # (e.g., smaller learning rates, better advantage estimations, regularization --> mb_advantages already doing normalization).
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    # ensuring that the more "penalizing" loss is used
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    # the 0.5 (1/2) is typically used for mathematical convenience when doing Mean Squared Error(MSE)
                    # MSE = 1/n * ∑ (Pi -Ti)^2 # P is the predict value, T is the truth value
                    # Taking the MSE derivative with respect to Pi
                    # ∂MSE/∂Pi = 1/n * 2 * (Pi - Ti)
                    # Without the 1/2 (0.5), you would have a factor of 2 in the gradient,
                    # which means you would need to divide the gradient update step by 2.
                    # Using 1/2 directly cancels this factor, making the gradient easier to handle.
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                # Monitoring `entropy_loss` in reinforcement learning algorithms like PPO is important for several reasons:
                ### Purpose of `entropy_loss`:
                # 1. **Encouraging Exploration**:
                #    - **Entropy** measures the randomness in the action distribution. High entropy indicates more exploration, while low entropy indicates a more deterministic policy.
                #    - By monitoring entropy, you ensure that the agent explores sufficiently, which is crucial for discovering optimal strategies.
                # 2. **Balancing Exploration and Exploitation**:
                #    - Entropy is often added as a regularization term in the loss function to keep the policy from becoming too deterministic too quickly.
                #    - This helps balance exploration (trying new actions) with exploitation (choosing the best-known actions).
                # 3. **Diagnostic Indicator**:
                #    - Monitoring `entropy_loss` can serve as an indicator of whether the agent is learning too aggressively or not exploring enough.
                #    - If entropy decreases too quickly, it might suggest that the agent is becoming overly confident in its actions and could get stuck in suboptimal policies.
                # 4. **Hyperparameter Tuning**:
                #    - The coefficient of the entropy term (`ent_coef`) can be adjusted to control the level of exploration. Monitoring `entropy_loss` can help in tuning this parameter effectively.

                # Overall, tracking `entropy_loss` aids in ensuring that the training process maintains a healthy level of exploration, which is critical for robust learning.
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Purpose: This line resets the gradients of all parameters to zero.
                # Significance in PPO: In neural network training, gradients accumulate by default in PyTorch.
                # After every backward pass, the gradients from the previous iteration are added to the current gradients.
                # By calling zero_grad(), we ensure that we start with a clean slate for gradients for each new optimizer step,
                # which is critical for effective learning.
                optimizer.zero_grad()
                # Purpose: This computes the gradients of the loss function with respect to the parameters of the model
                # (in this case, the agent's parameters).
                loss.backward()
                # Purpose: This function applies gradient clipping to prevent excessively large gradients from destabilizing training.
                # Significance in PPO: In reinforcement learning, particularly in deep reinforcement learning methods like PPO,
                # gradients can sometimes become very large, which may lead to unstable updates and cause the training to diverge.
                # By clipping the gradients to a specified maximum norm (args.max_grad_norm), we help stabilize the learning process,
                # making it safer and more robust against large updates that could happen with high variance rewards.
                #
                # Gradients = multi-variable partial derivatives
                # gradient = ∇f=( ∂f/∂x1, ∂f/∂x2, ... ∂f/∂xN),
                # partial derivative = ∂f/∂x1 with respect to x1, ∂f/∂x2 with respect to x2, ...
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # Explained Variance
        # it is the difference between the expected value and the predicted value
        # A machine learning model must have at least 60 percent of explained variance
        # refer to https://blog.csdn.net/YHKKun/article/details/137057987
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # SPS => steps per second
        # If SPS is lower than expected, it may indicate performance bottlenecks,
        # such as inefficient data handling, suboptimal hardware utilization, or other issues in your training loop.
        #
        # Additionally, tracking SPS over time gives insights into how changes to code or hardware configurations impact the speed of training.
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()

# Numerical Example
# Taking Q(s, a) ≈ Rt, with definition
# Rt=r_t + discount_factor*r_t+1 + discount_factor^2*r_t+2 + ...
# Q(s, a) = r_t + discount_factor*V_t+1
# Let’s consider a simplified environment where state s
# has two possible actions, a1 and a2. We can make assignments for the expected rewards associated with these actions:
# Assuming γ=0.9, an episode consists of 3 steps
# State s:
# Taking action a1 might yield:
# r1=5, r2=4, r3=3
# Calculating
# Q(s,a1):
# Q(s,a1) = r1+γ*r2+γ^2*r3 = 5+0.9*4+0.9^2*3 = 5+3.6+2.43 = 11.03

# Taking action a2 might yield:
# r1=3, r2=2, r3=1
# Calculating
# Q(s,a2):
# Q(s,a2) = r1+γ*r2+γ^2*r3 = 3+0.9*2+0.9^2*1 = 3+1.8+0.81 = 5.61

# Policy and Action Selection:

# Assume the current policy selects action a1 with a probability of 0.7 and action a2 with a probability of 0.3. This indicates:

# The value function
# V(s) can be calculated as the weighted average of the expected returns from all possible actions:
# V(s) = 0.7*Q(s, a1)+0.3*Q(s, a2) = 0.7*11.03+0.3*5.61 = 9.404

# Summary
# In this example, we find:

# V(s)is approximately 9.404.
# Q(s, a1) is 11.03.
# Q(s, a2) is 5.61.
# This demonstrates that:

# Difference: The Q-value for taking action a1(11.03) is higher than the expected value from the policy (9.404),
# indicating that the policy may not be optimal (since a1 has a higher payoff than what the policy suggests).

# For action a2, the Q-value (5.61) is lower than the policy's average (9.404), showing its disadvantages in the current policy context.
# Thus, the distinction between V(s) and Q(s, a) reflects the outcomes of policy selection,
# which is crucial information for the agent to improve its strategy.
