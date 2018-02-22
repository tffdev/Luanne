
--[[-------------------------------------------------------------------------
	Configurations
---------------------------------------------------------------------------]]--
-- alphabet matrix map
al_num = {}
alphabet="abcdefghijklmnopqrstuvwxyz, "
for i=1,#alphabet do
	al_num[i] = string.sub(alphabet, i, i)
end

-- Parameters
math.randomseed(123)
STRUCTURE 				= { #alphabet, 40, 40, 40, #alphabet}
learning_rate 			= 0.3
momentum_multiplier		= 0.1
iterations 				= 5000
epochs 					= 5
passes_through_file		= 1
directory 				= "NLPS/"
save_per 				= 1000

--[[
To:
	- Dataset changing
	- To do a full single backwards pass per letter

]]--