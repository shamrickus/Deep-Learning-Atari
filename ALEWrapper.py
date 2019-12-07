from ale_python_interface import ALEInterface

import BaseROM


class ALEWrapper:
    ale: ALEInterface = None

    def __init__(self):
        self.ale = ALEInterface()

    def set_seed(self, seed):
        self.ale.setInt(b'random_seed', seed)

    def set_display(self, display):
        self.ale.setBool(b'display_screen', display)

    def set_recording(self, sub_dir):
        dir = f"recording/{sub_dir}"
        self.ale.setString(b'record_screen_dir', str.encode(dir))
        self.ale.setInt(b'fragsize', 64)

    def set_repeat_action(self, probability=0.):
        self.ale.setFloat(b'repeat_action_probability', probability)

    def load_rom(self, rom: BaseROM):
        self.ale.loadROM(str.encode("roms/"+rom.name))

    def game_over(self):
        return self.ale.game_over()

    def get_minimal_actions(self):
        return list(self.ale.getMinimalActionSet())

    def act(self, action):
        return self.ale.act(action)

    def restore_state(self, state):
        self.ale.restoreState(state)

    def restore_system_state(self, state):
        self.ale.restoreSystemState(state)

    def copy_state(self):
        return self.ale.cloneState()

    def copy_system_state(self):
        return self.ale.cloneSystemState()

    def get_screen_dim(self):
        return self.ale.getScreenDims()

    def get_frame(self):
        return self.ale.getScreenRGB()

    def reset(self):
        self.ale.reset_game()

    def get_hashable_state(self, state):
        state.flags.writeable = False
        return state.ravel().data

    def map_action(self, action):
        if action == 0:
            return "noop"
        elif action == 1:
            return "fire"
        elif action == 2:
            return "up"
        elif action == 3:
            return "right"
        elif action == 4:
            return "left"
        elif action == 5:
            return "down"
        elif action == 6:
            return "up-right"
        elif action == 7:
            return "up-left"
        elif action == 8:
            return "down-right"
        elif action == 9:
            return "down-left"
        elif action == 10:
            return "up-fire"
        elif action == 11:
            return "right-fire"
        elif action == 12:
            return "left-fire"
        elif action == 13:
            return "down-fire"
        elif action == 14:
            return "up-right-fire"
        elif action == 15:
            return "up-left-fire"
        elif action == 16:
            return "down-right-fire"
        elif action == 17:
            return "down-left-fire"
        elif action == 40:
            return "reset"
