# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="_jQ1tEQCxwRx"
# ##### Copyright 2020 The TensorFlow Authors.

# + cellView="form" id="V_sgB_5dx1f1"
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# + [markdown] id="p62G8M_viUJp"
# # Playing CartPole with the Actor-Critic Method
#

# + [markdown] id="-mJ2i6jvZ3sK"
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic">
#     <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
#     View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
#     Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb">
#     <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
#     View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/reinforcement_learning/actor_critic.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# + [markdown] id="kFgN7h_wiUJq"
# This tutorial demonstrates how to implement the [Actor-Critic](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf) method using TensorFlow to train an agent on the [Open AI Gym](https://gym.openai.com/) CartPole-V0 environment.
# The reader is assumed to have some familiarity with [policy gradient methods](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) of reinforcement learning. 
#

# + [markdown] id="_kA10ZKRR0hi"
# **Actor-Critic methods**
#
# Actor-Critic methods are [temporal difference (TD) learning](https://en.wikipedia.org/wiki/Temporal_difference_learning) methods that represent the policy function independent of the value function. 
#
# A policy function (or policy) returns a probability distribution over actions that the agent can take based on the given state.
# A value function determines the expected return for an agent starting at a given state and acting according to a particular policy forever after.
#
# In the Actor-Critic method, the policy is referred to as the *actor* that proposes a set of possible actions given a state, and the estimated value function is referred to as the *critic*, which evaluates actions taken by the *actor* based on the given policy.
#
# In this tutorial, both the *Actor* and *Critic* will be represented using one neural network with two outputs.
#

# + [markdown] id="rBfiafKSRs2k"
# **CartPole-v0**
#
# In the [CartPole-v0 environment](https://gym.openai.com/envs/CartPole-v0), a pole is attached to a cart moving along a frictionless track. 
# The pole starts upright and the goal of the agent is to prevent it from falling over by applying a force of -1 or +1 to the cart. 
# A reward of +1 is given for every time step the pole remains upright.
# An episode ends when (1) the pole is more than 15 degrees from vertical or (2) the cart moves more than 2.4 units from the center.
#
# <center>
#   <figure>
#     <image src="images/cartpole-v0.gif">
#     <figcaption>
#       Trained actor-critic model in Cartpole-v0 environment
#     </figcaption>
#   </figure>
# </center>
#

# + [markdown] id="XSNVK0AeRoJd"
# The problem is considered "solved" when the average total reward for the episode reaches 195 over 100 consecutive trials.

# + [markdown] id="glLwIctHiUJq"
# ## Setup
#
# Import necessary packages and configure global settings.
#

# + id="13l6BbxKhCKp"
# !pip install -q gym

# + id="WBeQhPi2S4m5" language="bash"
# # Install additional packages for visualization
# sudo apt-get install -y xvfb python-opengl > /dev/null 2>&1
# pip install -q pyvirtualdisplay > /dev/null 2>&1
# pip install -q git+https://github.com/tensorflow/docs > /dev/null 2>&1

# + id="tT4N3qYviUJr"
import collections
import gym
import numpy as np
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


# Create the environment
env = gym.make("CartPole-v0")

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


# + [markdown] id="AOUCe2D0iUJu"
# ## Model
#
# The *Actor* and *Critic* will be modeled using one neural network that generates the action probabilities and critic value respectively. We use model subclassing to define the model. 
#
# During the forward pass, the model will take in the state as the input and will output both action probabilities and critic value $V$, which models the state-dependent [value function](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#value-functions). The goal is to train a model that chooses actions based on a policy $\pi$ that maximizes expected [return](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#reward-and-return).
#
# For Cartpole-v0, there are four values representing the state: cart position, cart-velocity, pole angle and pole velocity respectively. The agent can take two actions to push the cart left (0) and right (1) respectively.
#
# Refer to [OpenAI Gym's CartPole-v0 wiki page](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf) for more information.
#

# + id="aXKbbMC-kmuv"
class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)


# + id="nWyxJgjLn68c"
num_actions = env.action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)


# + [markdown] id="hk92njFziUJw"
# ## Training
#
# To train the agent, you will follow these steps:
#
# 1. Run the agent on the environment to collect training data per episode.
# 2. Compute expected return at each time step.
# 3. Compute the loss for the combined actor-critic model.
# 4. Compute gradients and update network parameters.
# 5. Repeat 1-4 until either success criterion or max episodes has been reached.
#

# + [markdown] id="R2nde2XDs8Gh"
# ### 1. Collecting training data
#
# As in supervised learning, in order to train the actor-critic model, we need
# to have training data. However, in order to collect such data, the model would
# need to be "run" in the environment.
#
# We collect training data for each episode. Then at each time step, the model's forward pass will be run on the environment's state in order to generate action probabilities and the critic value based on the current policy parameterized by the model's weights.
#
# The next action will be sampled from the action probabilities generated by the model, which would then be applied to the environment, causing the next state and reward to be generated.
#
# This process is implemented in the `run_episode` function, which uses TensorFlow operations so that it can later be compiled into a TensorFlow graph for faster training. Note that `tf.TensorArray`s were used to support Tensor iteration on variable length arrays.

# + id="5URrbGlDSAGx"
# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], 
                           [tf.float32, tf.int32, tf.int32])


# + id="a4qVRV063Cl9"
def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> List[tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)
  
    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)
  
    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])
  
    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)
  
    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()
  
  return action_probs, values, rewards


# + [markdown] id="lBnIHdz22dIx"
# ### 2. Computing expected returns
#
# We convert the sequence of rewards for each timestep $t$, $\{r_{t}\}^{T}_{t=1}$ collected during one episode into a sequence of expected returns $\{G_{t}\}^{T}_{t=1}$ in which the sum of rewards is taken from the current timestep $t$ to $T$ and each reward is multiplied with an exponentially decaying discount factor $\gamma$:
#
# $$G_{t} = \sum^{T}_{t'=t} \gamma^{t'-t}r_{t'}$$
#
# Since $\gamma\in(0,1)$, rewards further out from the current timestep are given less weight.
#
# Intuitively, expected return simply implies that rewards now are better than rewards later. In a mathematical sense, it is to ensure that the sum of the rewards converges.
#
# To stabilize training, we also standardize the resulting sequence of returns (i.e. to have zero mean and unit standard deviation).
#

# + id="jpEwFyl315dl"
def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

  return returns


# + [markdown] id="1hrPLrgGxlvb"
# ### 3. The actor-critic loss
#
# Since we are using a hybrid actor-critic model, we will use loss function that is a combination of actor and critic losses for training, as shown below:
#
# $$L = L_{actor} + L_{critic}$$
#
# #### Actor loss
#
# We formulate the actor loss based on [policy gradients with the critic as a state dependent baseline](https://www.youtube.com/watch?v=EKqxumCuAAY&t=62m23s) and compute single-sample (per-episode) estimates.
#
# $$L_{actor} = -\sum^{T}_{t=1} log\pi_{\theta}(a_{t} | s_{t})[G(s_{t}, a_{t})  - V^{\pi}_{\theta}(s_{t})]$$
#
# where:
# - $T$: the number of timesteps per episode, which can vary per episode
# - $s_{t}$: the state at timestep $t$
# - $a_{t}$: chosen action at timestep $t$ given state $s$
# - $\pi_{\theta}$: is the policy (actor) parameterized by $\theta$
# - $V^{\pi}_{\theta}$: is the value function (critic) also parameterized by $\theta$
# - $G = G_{t}$: the expected return for a given state, action pair at timestep $t$
#
# We add a negative term to the sum since we want to maximize the probabilities of actions yielding higher rewards by minimizing the combined loss.
#
# <br>
#
# ##### Advantage
#
# The $G - V$ term in our $L_{actor}$ formulation is called the [advantage](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#advantage-functions), which indicates how much better an action is given a particular state over a random action selected according to the policy $\pi$ for that state.
#
# While it's possible to exclude a baseline, this may result in high variance during training. And the nice thing about choosing the critic $V$ as a baseline is that it trained to be as close as possible to $G$, leading to a lower variance.
#
# In addition, without the critic, the algorithm would try to increase probabilities for actions taken on a particular state based on expected return, which may not make much of a difference if the relative probabilities between actions remain the same.
#
# For instance, suppose that two actions for a given state would yield the same expected return. Without the critic, the algorithm would try to raise the probability of these actions based on the objective $J$. With the critic, it may turn out that there's no advantage ($G - V = 0$) and thus no benefit gained in increasing the actions' probabilities and the algorithm would set the gradients to zero.
#
# <br>
#
# #### Critic loss
#
# Training $V$ to be as close possible to $G$ can be set up as a regression problem with the following loss function:
#
# $$L_{critic} = L_{\delta}(G, V^{\pi}_{\theta})$$
#
# where $L_{\delta}$ is the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss), which is less sensitive to outliers in data than squared-error loss.
#

# + id="9EXwbEez6n9m"
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss


# + [markdown] id="HSYkQOmRfV75"
# ### 4. Defining the training step to update parameters
#
# We combine all of the steps above into a training step that is run every episode. All steps leading up to the loss function are executed with the `tf.GradientTape` context to enable automatic differentiation.
#
# We use the Adam optimizer to apply the gradients to the model parameters.
#
# We also compute the sum of the undiscounted rewards, `episode_reward`, in this step which would be used later on to evaluate if we have met the success criterion.
#
# We apply the `tf.function` context to the `train_step` function so that it can be compiled into a callable TensorFlow graph, which can lead to 10x speedup in training.
#

# + id="QoccrkF3IFCg"
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode) 

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward


# + [markdown] id="HFvZiDoAflGK"
# ### 5. Run the training loop
#
# We execute training by run the training step until either the success criterion or maximum number of episodes is reached.  
#
# We keep a running record of episode rewards using a queue. Once 100 trials are reached, the oldest reward is removed at the left (tail) end of the queue and the newest one is added at the head (right). A running sum of the rewards is also maintained for computational efficiency. 
#
# Depending on your runtime, training can finish in less than a minute.

# + id="kbmBxnzLiUJx"
# %%time

max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

with tqdm.trange(max_episodes) as t:
  for i in t:
    initial_state = tf.constant(env.reset(), dtype=tf.float32)
    episode_reward = int(train_step(
        initial_state, model, optimizer, gamma, max_steps_per_episode))

    running_reward = episode_reward*0.01 + running_reward*.99
  
    t.set_description(f'Episode {i}')
    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)
  
    # Show average episode reward every 10 episodes
    if i % 10 == 0:
      pass # print(f'Episode {i}: average reward: {avg_reward}')
  
    if running_reward > reward_threshold:  
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

# + [markdown] id="ru8BEwS1EmAv"
# ## Visualization
#
# After training, it would be good to visualize how the model performs in the environment. You can run the cells below to generate a GIF animation of one episode run of the model. Note that additional packages need to be installed for OpenAI Gym to render the environment's images correctly in Colab.

# + id="qbIMMkfmRHyC"
# Render an episode and save as a GIF file

from IPython import display as ipythondisplay
from PIL import Image
from pyvirtualdisplay import Display


display = Display(visible=0, size=(400, 300))
display.start()


def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int): 
  screen = env.render(mode='rgb_array')
  im = Image.fromarray(screen)

  images = [im]
  
  state = tf.constant(env.reset(), dtype=tf.float32)
  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    action_probs, _ = model(state)
    action = np.argmax(np.squeeze(action_probs))

    state, _, done, _ = env.step(action)
    state = tf.constant(state, dtype=tf.float32)

    # Render screen every 10 steps
    if i % 10 == 0:
      screen = env.render(mode='rgb_array')
      images.append(Image.fromarray(screen))
  
    if done:
      break
  
  return images


# Save GIF image
images = render_episode(env, model, max_steps_per_episode)
image_file = 'cartpole-v0.gif'
# loop=0: loop forever, duration=1: play each frame for 1ms
images[0].save(
    image_file, save_all=True, append_images=images[1:], loop=0, duration=1)

# + id="TLd720SejKmf"
import tensorflow_docs.vis.embed as embed
embed.embed_file(image_file)

# + [markdown] id="lnq9Hzo1Po6X"
# ## Next steps
#
# This tutorial demonstrated how to implement the actor-critic method using Tensorflow.
#
# As a next step, you could try training a model on a different environment in OpenAI Gym. 
#
# For additional information regarding actor-critic methods and the Cartpole-v0 problem, you may refer to the following resources:
#
# - [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)
# - [Actor Critic Lecture (CAL)](https://www.youtube.com/watch?v=EKqxumCuAAY&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=7&t=0s)
# - [Cartpole learning control problem \[Barto, et al. 1983\]](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf) 
#
# For more reinforcement learning examples in TensorFlow, you can check the following resources:
# - [Reinforcement learning code examples (keras.io)](https://keras.io/examples/rl/)
# - [TF-Agents reinforcement learning library](https://www.tensorflow.org/agents)
#
