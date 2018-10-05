# Extensions to OpenAI gym

This repository is my own personal sandbox for playing with reinforcement learning algorithms. The reason why I started this is that I noticed that every time I wrote a little quick-n-dirty implementation of an RL algorithm, the code was so horrific and my results essentially irreproducible. So I figured that in order to advance and grow my RL knowledge, my code had to be organized.


**Disclaimer:** Although the code in here should be reasonably clean and bug-free, it's by no means complete.


## Organisation of Code

The way I chose to organize the code is that I defined *value functions*, *policies* and *algorithms*. This is particularly useful for RL algorithms like Q-learning and I hope that other methods like policy-gradient methods will fit in this mold as well.


The central object *Q(s, a)* is defined as an abstract class `BaseQ`, which is refined in a variety of specific implementations, such as

- `TabularQ`
- `LinearQ` (implemented)
- `NeuralNetQ`

The policy (or policies) derived from such Q-functions is implemented as a wrapper class:

- `PolicyQ`

In order to update the state-action value function, we use algorithm objects, such as

- `Sarsa`
- `QLearning`

There are also state-only value functions such as `LinearV`, which can be updated e.g. using the `ValueTD0` algorithm.


## A Simple Example:

```python
import gym
from gym_extensions.value_functions import LinearQ
from gym_extensions.policies import PolicyQ
from gym_extensions.algorithms import Sarsa


env = gym.make('CartPole-v0')
q = LinearQ(env, interaction_degree=2)
p = PolicyQ(q)
algo = Sarsa(q, alpha=0.01, gamma=0.8)

for episode in range(1000):
    # init
    s = env.reset()
    a = env.action_space.sample()

    for t in range(200):
        # env.render()
        s_next, r, done, _ = env.step(a)
        a_next = p.greedy(s_next)
        algo.update(s, a, r, s_next, a_next)

        if done:
            break

        s, a = s_next, a_next

env.close()
```

For a more elaborate examples, check out the [notebooks](./notebooks).
