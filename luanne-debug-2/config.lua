
--[[-------------------------------------------------------------------------
	Configurations
---------------------------------------------------------------------------]]

-- Window
love.window.setMode(1200, 680, {resizable=true, vsync=true,fullscreen=false})

-- Datasets
inp = {{0,0},{0,1},{1,1},{1,0}}
-- exp_out = {{0},{1},{0},{1}}
exp_out = {{0,0},{0,1},{0,0},{0,1}}

-- Parameters
math.randomseed(123)
STRUCTURE 				= { 2, 7, 7, 2}
learning_rate 			= 0.1
momentum_multiplier		= 0.4
iterations_per_frame 	= 1
updates_per_sec 		= 60