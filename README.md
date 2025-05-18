# Q-Learning Controller for PySuperTuxKart

This repository contains a simple yet effective Q-learning implementation to train an autonomous driving agent in PySuperTuxKart (via the `pystk` Python bindings). The agent learns a discrete steering policy using only the x-coordinate of the kart's aim point on the track.

## Features

* **Tabular Q-Learning** with reward shaping
* **Discretized State Space**: only three buckets for steering direction (`-1`, `0`, `1`)
* **ε-Greedy Exploration** during training
* **Automatic Saving/Loading** of Q-tables in CSV format
* **Visualization** of cumulative rewards per episode
* **Support for Training & Evaluation** modes

## Requirements

* Python 3.7+
* [pystk (PySuperTuxKart Python bindings)](https://pystk.readthedocs.io)
* NumPy
* Matplotlib

Install dependencies via:

```bash
pip install numpy matplotlib pystk
```

## Repository Structure

* `qlearning.py` (main script & classes)
* `q_tables/` (directory for saved Q-tables)
* `plots/` (directory for training reward plots)

## Usage

Run the script with one or more track names:

```bash
python qlearning.py TRACK_NAME [TRACK_NAME ...] [options]
```

### Common Options

| Flag                 | Description                                             |
| -------------------- | ------------------------------------------------------- |
| `-m, --max-frames`   | Max frames per episode (default: 2500)                  |
| `-n, --num-training` | Number of training episodes (0 to skip training)        |
| `--epsilon`          | Exploration rate during training (default: 0.1)         |
| `--alpha`            | Learning rate α  (default: 0.1)                         |
| `--gamma`            | Discount factor γ  (default: 0.9999)                    |
| `-p, --planner`      | Enable planner model for steering predictions (default) |
| `--no-verbose`       | Disable rendering & debug plots during evaluation       |

## Key Components

### `QLearningController`

Manages the Q-table and action selection.

* **Constructor parameters:**

  * `alpha` (float): learning rate
  * `gamma` (float): discount factor
  * `epsilon` (float): exploration rate
  * `step_penalty` (float): per-step reward penalty
  * `finish_reward` (float): bonus for completing the race

* **Methods:**

  * `reset()`: clear episode history
  * `__call__(aim_point, current_vel, progress)`: select steering action and update Q-table
  * `update(state, action, next_state, reward)`: Q-table update rule
  * `save(track_name, directory)`: write Q-values to CSV
  * `load(track_name, directory)`: load Q-values from CSV

### `PyTux`

Wraps `pystk` for consistent rollouts.

* **Initialization:**

  * `screen_width`, `screen_height`: render resolution
  * `train` (bool): use minimal graphics during training

* **Methods:**

  * `rollout(track, controller, ...)`: run a single episode (training or eval)
  * `close()`: clean up and close renderer

## Example Workflow

1. **Train an Agent** on the `cornfield_crossing` track:

   ```bash
   python qlearning.py cornfield_crossing -n 1000 --epsilon 0.2
   ```

   * Keeps updating the Q-table and saves:

     * `plots/cornfield_crossing_cumulative_rewards.png`
     * `q_tables/cornfield_crossing_Q.csv`

2. **Evaluate the Agent** (no exploration):

   ```bash
   python qlearning.py cornfield_crossing -n 0 --no-verbose
   ```

   * Loads the saved Q-table and reports steps & progress.

## License

This project is released under the MIT License. Feel free to modify and extend!
