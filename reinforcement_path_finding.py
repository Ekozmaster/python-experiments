import numpy as np

spawn_pos = np.array([0, 0])
goal_pos = np.array([3, 2])
danger_pos = np.array([3, 1])


code_to_dir = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]


class Arena:
    grid = np.zeros((3, 4))

    def __init__(self):
        self.grid[goal_pos[1], goal_pos[0]] = 10
        self.grid[danger_pos[1], danger_pos[0]] = -10

    def get_value(self, x, y):
        return self.grid[y, x]

    def set_value(self, x, y, value):
        self.grid[y, x] = value

    @staticmethod
    def get_available_moves(x, y):
        available_moves = [0, 1, 2, 3]
        if y == 0 or (y == 2 and x == 1):
            available_moves.remove(2)
        if y == 2 or (y == 0 and x == 1):
            available_moves.remove(0)
        if x == 0 or (y == 1 and x == 2):
            available_moves.remove(3)
        if x == 3 or (y == 1 and x == 0):
            available_moves.remove(1)
        return available_moves

    @staticmethod
    def get_exploration(position):
        available_moves = Arena.get_available_moves(position[0], position[1])
        return position + code_to_dir[available_moves[np.random.randint(0, len(available_moves))]]

    def get_exploitation(self, position):
        available_moves = Arena.get_available_moves(position[0], position[1])
        if len(available_moves):
            pos = position + code_to_dir[available_moves[0]]
            best_move_value = self.get_value(pos[0], pos[1])
            best_move_code = available_moves[0]
            for move in available_moves:
                pos = position + code_to_dir[move]
                val = self.get_value(pos[0], pos[1])
                if val > best_move_value:
                    best_move_value = val
                    best_move_code = move

            return position + code_to_dir[best_move_code]
        return position


def get_environment_reward(position):
    if np.array_equal(position, goal_pos):
        return 10, True
    if np.array_equal(position, danger_pos):
        return -10, True
    return -1, False


def train_for_episode(position, arena, exploration_rate=1.0, discount_power=1, learning_rate=1.0):
    learning_rate *= 0.99
    exploration_rate *= 0.99
    explore_chance = np.random.rand()

    if explore_chance < exploration_rate:
        position = arena.get_exploration(position)
    else:
        position = arena.get_exploitation(position)

    reward, finish_episode = get_environment_reward(position)

    if finish_episode or discount_power > 1000:
        return reward
    future_reward = train_for_episode(position, arena, exploration_rate, discount_power + 1, learning_rate)
    discount_factor = pow(0.95, discount_power)
    learned_value = reward + discount_factor * future_reward

    old_value = arena.get_value(position[0], position[1])
    new_value = (1-learning_rate) * old_value + learning_rate*learned_value
    arena.set_value(position[0], position[1], new_value)
    return learned_value


arena = Arena()
print(np.flipud(arena.grid))
for i in range(10000):
    train_for_episode(np.array([0, 0]), arena, 1.0)
    print(np.flipud(arena.grid))
