from vizdoom import DoomGame
from vizdoom import ScreenResolution
from vizdoom import ScreenFormat
from vizdoom import GameVariable

def init(buttons):
    game = DoomGame()
    game.set_vizdoom_path("../../../ViZDoom/bin/vizdoom")
    game.set_doom_game_path("../../../ViZDoom/scenarios/freedoom2.wad")
    game.set_doom_scenario_path("../../../ViZDoom/scenarios/basic.wad")
    game.set_doom_map("map01")
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)

    # Sets other rendering options
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)

    # Adds buttons that will be allowed.
    for button in buttons:
        game.add_available_button(button)


    # Adds game variables that will be included in state.
    game.add_available_game_variable(GameVariable.AMMO2)
    game.add_available_game_variable(GameVariable.SELECTED_WEAPON)

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(300)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    game.set_window_visible(True)

    # Turns on the sound. (turned off by default)
    game.set_sound_enabled(True)

    # Sets the livin reward (for each move) to -1
    game.set_living_reward(-1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(Mode.PLAYER)

    # Initialize the game. Further configuration won't take any effect from now on.
    # game.set_console_enabled(True)
    game.init()