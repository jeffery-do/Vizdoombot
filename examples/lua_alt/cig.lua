require 'sys'

local vizdoom = require("../../src/lib_lua/init.lua")

local Button = vizdoom.Button
local Mode = vizdoom.Mode
local GameVariable = vizdoom.GameVariable
local ScreenFormat = vizdoom.ScreenFormat
local ScreenResolution = vizdoom.ScreenResolution

-- Create DoomGame instance. It will run the game and communicate with you.
local game = vizdoom.ViZDoomLua()

-- Use CIG example config or your own.
game:loadConfig("../../examples/config/cig.cfg")

-- Select game and map you want to use.
game:setDoomGamePath("../../scenarios/freedoom2.wad")
-- Not provided with environment due to licences
-- game:setDoomGamePath("../../scenarios/doom2.wad")

game:setDoomMap("map01") -- Limited deathmatch.
--game:setDoomMap("map02") -- Full deathmatch.

-- Join existing game.
game:addGameArgs("-join 127.0.0.1") -- Connect to a host for a multiplayer game.

-- Name your agent and select color
-- colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game:addGameArgs("+name AI +colorset 0")

-- Multiplayer requires the use of asynchronous modes.
game:setMode(Mode.ASYNC_PLAYER);

--game:setWindowVisible(false)

game:init();

-- Three example sample actions
local actions = {
   [1] = torch.IntTensor({1,0,0,0,0,0,0,0,0}),
   [2] = torch.IntTensor({0,1,0,0,0,0,0,0,0}),
   [3] = torch.IntTensor({0,0,1,0,0,0,0,0,0})
}

-- To be used by the main game loop
local s, reward

--------------------------------------------------------------------------------
-- Main game loop

-- Play until the game (episode) is over.
while not game:isEpisodeFinished() do

   if game:isPlayerDead() then
      -- Respawn immediately after death, new state will be available.
      game:respawnPlayer()
   end

   -- Analyze the state
   s = game:getState()

   -- Make a random action
   local action = actions[torch.random(#actions)]
   reward = game:makeAction(action)

    print("Frags:", game:getGameVariable(GameVariable.FRAGCOUNT))
end

game:close()
