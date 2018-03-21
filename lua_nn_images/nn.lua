--[[

Micro Neural Net Framework [For Project 'NLPS', Daniel Brier]
--------------------------------
Synapse indexing is as follows:
	syns[weight layer][going_to][coming_from]

]]--

-- Parameters
math.randomseed(123)

-- Requirements
reqs = {"funcs","nnfuncs","argcheck","config","tablesave"}
for i=1,#reqs do require(reqs[i]) end

json = require("json")

-- Declarations/inits
syns 		= {}
nodes 		= {}
gamma 		= {}
outputs 	= {}
it_count	= 0
syns 		= createStructure(STRUCTURE)
inp 		= {{}}
exp_out 	= {{}}
config_data = {step=0}
avg_error 	= 0


-- Load previous weights, if any
if(file_exists("synapses")) then
	syns = table.load("synapses")
	config_data = table.load("data")
end

if(inargs("-learn"))then
	-- Loop for forever, until you cancel 
	while(true) do
		-- Generate data to pass through
		config_data.step = config_data.step+1
		if(config_data.step>=2500) then config_data.step = 0 end

		-- DO PASS ON A CAT
			input_file = table.load("ml_json_cats/"..config_data.step)
			-- result is now vector table
			inp[1] = input_file
			-- print_r(inp)
			exp_out = {{1,0}}
			changes_matrix = createStructure(STRUCTURE, true)
			fullpass()

			syns = subweights(syns,changes_matrix)

		-- DO PASS ON A DOG
			input_file = table.load("ml_json_dogs/"..config_data.step)
			-- result is now vector table
			inp[1] = input_file
			-- print_r(inp)
			exp_out = {{0,1}}
			changes_matrix = nil
			changes_matrix = createStructure(STRUCTURE, true)
			fullpass()

			syns = subweights(syns,changes_matrix)

			-- One iteration is both a cat and dog pass. 2 full passes thorugh the network
		it_count = it_count + 1
		avg_error = avg_error + errs/#inp

		-- When the status printing criteria is met
		if(it_count == save_per) then

			-- Tell the user what the error is
			print("Average Error over dataset: "..avg_error/save_per,"(Weights saved)","Step "..config_data.step)

			-- Save the synapse tables
			table.save(syns,"synapses")
			table.save(config_data,"data")

			-- Reset stats
			avg_error = 0
			it_count = 0

			-- reload configurations
			require("config")

			-- check if status is set to 0, if so, stop learning
			-- Used to make sure not to cancel within loops becuase
			-- messes up the synapses ;;
			local file = io.open("status","rw")
			if(file:read()=="0") then
				break;
			end
		end
	end
	print("Finished learning")
else

	local result = table.load("ml_json_cats/"..902)

	-- result is now vector table of 900 len
	inp[1] = result

	result = forward(inp[1])
	print_r(result)
	if(result[1]>result[2])then
		print("It's a cat!")
	else
		print("It's a dog!")
	end
end