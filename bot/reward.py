from vizdoom import GameVariable

class RewardCalculator():
    def __init__(self):
        self.running_total = 0

    def calc_reward(self, game):
        # Assume Action Performed
        cur_reward = -5

        # Kills
        cur_killcount = game.get_game_variable(GameVariable.KILLCOUNT)
        new_kills = cur_killcount - self.prev_killcount
        if new_kills > 0:
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            cur_reward += 1000 * new_kills

        # Health
        cur_health = game.get_game_variable(GameVariable.HEALTH)
        diff_health = cur_health - self.prev_health
        if diff_health > 0:
            cur_reward += 10 * diff_health
        elif diff_health < 0:
            cur_reward += 20 * diff_health

        # Ammo
        cur_ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        diff_ammo = cur_ammo - self.prev_ammo
        if diff_ammo > 0:
            cur_reward += 10 * diff_ammo
        elif diff_ammo < 0:
            cur_reward += 100 * diff_ammo


        # Store This State
        self.prev_killcount = cur_killcount
        self.prev_health = cur_health
        self.prev_ammo = cur_ammo

        # Return Running Total
        self.running_total += cur_reward
        return cur_reward

    def get_total_reward(self):
        return self.running_total

    def reset(self, game):
        self.prev_killcount = game.get_game_variable(GameVariable.KILLCOUNT)
        self.prev_health = game.get_game_variable(GameVariable.HEALTH)
        self.prev_ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.running_total = 0