
--[[-------------------------------------------------------------------------
	Configurations
---------------------------------------------------------------------------]]

-- Window
love.window.setMode(1200, 680, {resizable=true, vsync=true,fullscreen=false})

-- Datasets
inp = {{0,0},{0,1},{1,1},{1,0}}
exp_out = {{0,0},{0,1},{0,0},{0,1}}

-- Parameters
math.randomseed(123)
STRUCTURE 				= { 2, 10, 10, 2}
learning_rate 			= 2
momentum_multiplier		= 0.5
iterations_per_frame 	= 50
updates_per_sec 		= 60