import os
import csv
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pystk

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15



class QLearningController:
    """
    Q-learning based controller for PyTux.
    Action space: steer in {-1, 0, 1}.
    State: discretized aim_point x-coordinate only.
    Reward:
      - small constant penalty each step
      - bonus/penalty based on whether last action and aim_point[0] share sign
      - large finish bonus when race completes
    """
    def __init__(self, alpha=0.1, gamma=0.9999, epsilon=0.1,
                 step_penalty=-0.1, finish_reward=1000.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [-1, 0, 1]
        self.Q = Counter()
        self.prev_state = None
        self.prev_action = None
        self.prev_progress = 0.0
        self.step_penalty = step_penalty
        self.finish_reward = finish_reward
        self.cum_reward =  []

    def reset(self):
        """Reset episode-specific state."""
        self.prev_state = None
        self.prev_action = None
        self.prev_progress = 0.0
        self.cum_reward = []

    def getState(self, aim_point, current_vel, progress):
        """
        Discretize the continuous observation into a state.
        aim_point: [x, y] in [-1,1]
        current_vel: float (unused)
        progress: float between 0 and 1 indicating progress on track (unused)
        
        Returns the aim_bucket (-1, 0, or 1)
        """
        # Discretize aim_point.x based on sign
        aim_x = aim_point[0]
        if aim_x < 0:
            aim_bucket = -1
        elif aim_x > 0:
            aim_bucket = 1
        else:
            aim_bucket = 0
            
        return aim_bucket
        

    def add_finish_reward(self):
        #print(15*"#" + "Got finish reward" + 15*"#")
        #print('before:', sum(self.cum_reward))
        reward = self.finish_reward * (self.gamma ** len(self.cum_reward) )
        self.cum_reward.append(reward)
        #print('after', sum(self.cum_reward))

        terminal_state = -2
        self.update(self.prev_state, self.prev_action, terminal_state, reward)

    def __call__(self, aim_point, current_vel, progress):
        state = self.getState(aim_point, current_vel, progress)

        if self.prev_state is not None:
            # 1) per-step penalty
            reward = self.step_penalty

            # Get the current action for reward calculation
            if random.random() < self.epsilon:
                action = random.choice(self.actions)
            else:
                q_vals = [self.Q[(state, a)] for a in self.actions]
                max_q = max(q_vals)
                best = [a for a, q in zip(self.actions, q_vals) if q == max_q]
                action = random.choice(best)

            # 2) directional reward / penalty based on state and action
            # state is now directly the aim_bucket (-1, 0, 1)
            aim_bucket = state
            if (aim_bucket == -1 and action == -1) or (aim_bucket == 1 and action == 1) or (aim_bucket == 0 and action == 0):
                reward += 1  # action aligns with aim_bucket direction
            else:
                reward -= 1  # action doesn't align with aim_bucket direction

            # accumulate and update
            self.cum_reward.append(reward * (self.gamma ** len(self.cum_reward)) )
            if self.epsilon > 0: # only update in training
                self.update(self.prev_state, self.prev_action, state, reward)

        # ε-greedy action selection
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_vals = [self.Q[(state, a)] for a in self.actions]
            max_q = max(q_vals)
            best = [a for a, q in zip(self.actions, q_vals) if q == max_q]
            action = random.choice(best)

        # store for next iteration
        self.prev_state = state
        self.prev_action = action
        self.prev_progress = progress

        # build the actual PyTux action
        act = pystk.Action()
        act.steer        = action
        act.acceleration = 1
        act.drift        = abs(aim_point[0]) > 0.4
        act.nitro        = (current_vel < 0.4)
        act.brake        = act.drift and act.nitro
        return act

    def update(self, state, action, next_state, reward):
        # if next_state is terminal (None or -2), future_q = 0
        future_q = 0.0 if next_state is None or next_state == -2 else max(self.Q[(next_state, a)] for a in self.actions)
        current_q = self.Q[(state, action)]
        self.Q[(state, action)] = current_q + \
            self.alpha * (reward + self.gamma * future_q - current_q)

    def save(self, track_name, directory='q_tables'):
        """Save the Q-table to a CSV file named <directory>/<track_name>_Q.csv."""
        os.makedirs(directory, exist_ok=True)
        fn = os.path.join(directory, f'{track_name}_Q.csv')
        with open(fn, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["State", "Action", "Q-value"])
            for (state, action), q_value in self.Q.items():
                writer.writerow([state, action, q_value])
        print(f'→ Q-table saved to {fn}')

    def load(self, track_name, directory='q_tables'):
        """Load the Q-table from a CSV file named <directory>/<track_name>_Q.csv."""
        fn = os.path.join(directory, f'{track_name}_Q.csv')
        if os.path.isfile(fn):
            self.Q = Counter()
            with open(fn, mode='r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    state = eval(row[0])  # restore tuple
                    action = int(row[1])
                    q_value = float(row[2])
                    self.Q[(state, action)] = q_value
            print(f'← Loaded Q-table from {fn}')
            return True
        else:
            print(f'! No Q-table found at {fn}, starting fresh')
            return False


class PyTux:
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96, train=False):
        assert PyTux._singleton is None, "Cannot create more than one PyTux object"
        PyTux._singleton = self
        if train:
            self.config = pystk.GraphicsConfig.none()
        else:
            self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def rollout(self, track, controller, planner=None,
                max_frames=2500, verbose=False, data_callback=None):
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track_obj = pystk.Track()
        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

        for t in range(max_frames):
            state.update()
            track_obj.update()
            kart = state.players[0].kart

            # check for finish
            if np.isclose(kart.overall_distance / track_obj.length, 1.0, atol=3e-3):
                if controller.epsilon > 0:
                    controller.add_finish_reward() # we should add finish reward here
                if verbose:
                    print(f"Finished at t={t}")
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T
            aim_world = self._point_on_track(
                kart.distance_down_track + TRACK_OFFSET, track_obj)
            aim_image = self._to_image(aim_world, proj, view)
            current_vel = np.linalg.norm(kart.velocity)
            progress = kart.overall_distance / track_obj.length

            if data_callback is not None:
                data_callback(t, np.array(self.k.render_data[0].image), aim_image)

            if planner:
                image = np.array(self.k.render_data[0].image)
                aim_image = planner(image)

            action = controller(aim_image, current_vel, progress)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(plt.Circle(
                    WH2 * (1 + self._to_image(kart.location, proj, view)),
                    2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(
                    WH2 * (1 + aim_image), 2, ec='r', fill=False, lw=1.5))
                plt.pause(1e-3)

            self.k.step(action)

        return t, kart.overall_distance / track_obj.length

    def close(self):
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-m', '--max-frames', type=int, default=2500,
                        help='Max frames per episode')
    parser.add_argument('-n', '--num-training', type=int, default=0,
                        help='Number of training episodes (0 to skip training)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration rate during training')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9999,
                        help='Discount factor')
    parser.add_argument('-p', '--planner', action='store_true', default=True)
    parser.add_argument(
    '--no-verbose',
    dest='verbose',
    action='store_false',
    help='disable verbose output'
)
# verbose defaults to True


    args = parser.parse_args()

    if args.planner:
        print("Loaded planner!")
        #planner = load_model().eval()
    else:
        planner = None


    pytux = PyTux(train=(args.num_training > 0))
    agent = QLearningController(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        step_penalty=-1,
        finish_reward=1000.0
    )

    for track in args.track:
        print(f'=== TRACK: {track} ===')

        # 1. Train (if requested)
        if args.num_training > 0:
            cum_rewards = []
            for episode in range(args.num_training):
                if episode % 100 == 0: print(f'  Training episode {episode+1}/{args.num_training}')
                agent.reset()
                pytux.rollout(track, agent, max_frames=args.max_frames)
                if episode % 100 == 0: print(f'    → Episode cumulative reward: {sum(agent.cum_reward):.2f}')
                cum_rewards.append(sum(agent.cum_reward))

            # Plot cumulative reward per episode
            fig, ax = plt.subplots()
            ax.plot(range(1, args.num_training+1), cum_rewards)
            #ax.scatter(range(1, args.num_training+1), cum_rewards, marker='o', s=20, alpha=0.7)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Cumulative Reward per Episode for {track}')
            #plt.show()

            # Save plot to file
            plots_dir = 'plots'
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, f'{track}_cumulative_rewards.png')
            fig.savefig(plot_path)
            print(f'→ Plot saved to {plot_path}')

            # Save Q-table
            agent.save(track)

        else:
            # 2. Load existing Q-table (if any)
            agent.load(track)

            # 3. Evaluate with no exploration
            agent.epsilon = 0.0
            agent.reset()
            steps, prog = pytux.rollout(
                track, agent, max_frames=args.max_frames, verbose=args.verbose)
            print(f'  → Eval on {track}: steps={steps}, progress={prog:.3f}')
            print(f'    → Eval cumulative reward: {sum(agent.cum_reward):.2f}')

    pytux.close()
