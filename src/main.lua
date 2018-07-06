--[[

Micro Neural Net Framework [For Project 'NLPS', Daniel Brier]
--------------------------------
Synapse indexing is as follows:
	syns[weight layer][going_to][coming_from]

]]--

-- Parameters
math.randomseed(123)

-- Requirements
require("lib/funcs")
require("lib/nnfuncs")
require("deps/tablesave")
local Bitmap = require("deps/bitmap")

local savestate_name = arg[2]
local loaded_config = table.load("../savestates/"..savestate_name.."/config")
if(loaded_config == nil) then
	print("Project "..savestate_name.." does not exist!")
	os.exit()
end
-- To modify based on config
STRUCTURE 				= {900, 18, 18, #loaded_config["images"]}
learning_rate 			= tonumber(loaded_config["learning-rate"])
momentum_multiplier		= tonumber(loaded_config["momentum"])
save_per 				= tonumber(loaded_config["save-per-count"])


-- Declarations/inits
syns 		= {}
nodes 		= {}
gamma 		= {}
outputs 	= {}
it_count	= 0
inp 		= {{}}
exp_out 	= {{}}
config_data = {step=0}
avg_error 	= 0

if( not file_exists("../savestates/"..savestate_name.."/config")) then
	print("project "..savestate_name.." does not exist!")
	os.exit()
end

config_data = table.load("../savestates/"..savestate_name.."/config")

if(config_data == nil) then 
	print("config_data nil!") 
end

-- Load previous weights, if any
if(file_exists("../savestates/"..savestate_name.."/synapses")) then
	syns = table.load("../savestates/"..savestate_name.."/synapses")
else
	print("Created new synapse database!")
	syns = createStructure(STRUCTURE)
	table.save(syns, "../savestates/"..savestate_name.."/synapses")
end

function main()
	if(arg[1] == "learn")then
		-- Loop for forever, until you cancel 
		learningLoop()
		print("Finished learning")
	elseif(arg[1] == "do")then

		local img = Bitmap.from_file(arg[3])
		if(not img) then
			print("`"..arg[3].."` is not a valid bitmap")
			os.exit()
		end

		local output_vector = {}
		local fail = false
		for i=1,30 do
			for j=1,30 do
				local pixel = {img:get_pixel(i,j)}
				if(pixel[1] and pixel[2] and pixel[3]) then
					table.insert(output_vector,((pixel[1]+pixel[2]+pixel[3])/3)/255)
				else
					fail = true
				end
			end
		end

		local result = output_vector
		-- result is now vector table of 900 len
		inp[1] = result
		result = forward(inp[1])
		print_r(result)
		local largest = {1,result[1]}
		for i=2,#result do
			if(result[i] > largest[2]) then
				largest[1] = i
				largest[2] = result[i]
			end
		end
		print("Category: "..loaded_config["images"][largest[1]][2])
	end
end

function learningLoop()
	config_data.step = config_data.step or 1
	while(true) do
		-- Generate data to pass through
		config_data.step = config_data.step + 1 

		if(config_data.step >= 500) then config_data.step = 0 end

		for i=1,#loaded_config["images"] do
			-- DO PASS ON A CAT
			input_file = table.load("../savestates/"..savestate_name.."/"..loaded_config["images"][i][2].."/"..config_data.step)
			if(input_file ~= nil) then
				-- result is now vector table
				inp[1] = input_file
				-- print_r(inp)
				exp_out = {{0,0}}
				exp_out[1][i] = 1
				--[[ 
					PERFORM PASS ON NETWORK
					"fullpass" performs a pass on globals, and then
					returns the mean squared error
				--]] 
				avg_error = avg_error + fullpass()/#inp

				syns = subweights(syns,changes_matrix)
			end
		end

		-- One iteration is both a cat and dog pass. 2 full passes thorugh the network
		it_count = it_count + 1
		-- print(avg_error)
		-- When the status printing criteria is met
		if(it_count == save_per) then
			-- Tell the user what the error is
			print("Average Error over dataset: "..tonumber(avg_error/it_count),"(Weights saved)","Step "..config_data.step)
			saveSynapses()
		end
	end
end

function saveSynapses()
	-- Save the synapse tables
	table.save(syns,"../savestates/"..savestate_name.."/synapses")
	table.save(config_data,"../savestates/"..savestate_name.."/config")

	-- Reset stats
	avg_error = 0
	it_count = 0

	if(getcontent("status") == "0")then
		print("Safely exiting...")
		os.exit()
	end
end

main()