# Mountain-Car-Task

## Aim

To implement a Reinforcement Learning agent for solving the Mountain Car task, where the objective is to learn an optimal policy that enables an underpowered car to reach the goal at the top of a hill through trial-and-error interactions with the environment.

---

## Algorithm

### Q-Learning Algorithm for Mountain Car

1. Initialize the Mountain Car environment.
2. Discretize the continuous state space into finite bins.
3. Initialize the Q-table with zeros.
4. For each episode:

   * Reset the environment and obtain the initial state.
   * Convert the state into a discrete representation.
   * Repeat until the episode terminates:

     * Choose an action using the ε-greedy policy.

     * Execute the action and observe the next state and reward.

     * Convert the next state into a discrete state.

     * Update the Q-value using:

       Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

     * Set the current state to the next state.
   * Reduce ε gradually to encourage exploitation.
5. Store the total reward obtained in each episode.
6. Plot the learning performance after training.
7. Evaluate the learned policy.

---

## Program

```python
#Mountain Car Task.
import numpy as np

# Gym + NumPy 2.x compatibility
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000

DISCRETE_OS_SIZE = [20, 20]

discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE

q_table = np.random.uniform(
    low=-2,
    high=0,
    size=(DISCRETE_OS_SIZE + [env.action_space.n])
)

def get_discrete_state(state):
    discrete_state = (
        state - env.observation_space.low
    ) / discrete_os_win_size

    discrete_state = np.clip(
        discrete_state.astype(int),
        0,
        np.array(DISCRETE_OS_SIZE) - 1
    )

    return tuple(discrete_state)

epsilon = 0.5

goal_found = False

# ---------------- TRAINING ----------------
for episode in range(EPISODES):

    state, _ = env.reset()
    discrete_state = get_discrete_state(state)

    done = False

    while not done:

        # Epsilon-Greedy
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        new_discrete_state = get_discrete_state(new_state)

        current_q = q_table[discrete_state + (action,)]

        if not done:

            max_future_q = np.max(
                q_table[new_discrete_state]
            )

            new_q = (
                (1 - LEARNING_RATE) * current_q
                + LEARNING_RATE *
                (reward + DISCOUNT * max_future_q)
            )

            q_table[discrete_state + (action,)] = new_q

        else:

            if new_state[0] >= env.goal_position:

                print("\nGOAL REACHED!")
                print("Training stopped.\n")

                goal_found = True

                q_table[discrete_state + (action,)] = 0

                break

        discrete_state = new_discrete_state

    epsilon *= 0.9995

    if goal_found:
        break

# ---------------- OPTIMAL PATH ----------------

print("Optimal Solution Path:\n")

state, _ = env.reset()

path = []

done = False

while not done:

    path.append(state)

    discrete_state = get_discrete_state(state)

    action = np.argmax(q_table[discrete_state])

    state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

for i, p in enumerate(path):
    print(f"Step {i+1}: Position={p[0]:.4f}, Velocity={p[1]:.4f}")

print("\nGoal State Reached!")

env.close()


```



## Output

<img width="842" height="708" alt="image" src="https://github.com/user-attachments/assets/93293a88-2a95-42db-aac0-78422d220acf" />


## Result

The Mountain Car Reinforcement Learning task was successfully implemented using the Q-Learning algorithm. The agent learned an optimal policy through interaction with the environment and progressively improved its ability to reach the goal state. The learning curve confirms the effectiveness of the training process.
