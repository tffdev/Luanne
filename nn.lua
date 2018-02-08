--[[

Micro Neural Net Framework [For Project 'NLPS', Daniel Brier]
--------------------------------
Synapse indexing is as follows:
	syns[weight layer][going_to][coming_from]
]]--

-- Requirements
reqs = {"json","funcs","nnfuncs","argcheck"}
for i=1,#reqs do require(reqs[i]) end

-- Declarations
	syns = {}
	layers = {}
	gammas = {}
	math.randomseed(123)

-- DATASET
	inp = {{0,0},{0,1},{1,1},{1,0}}
	exp_out = {{0,0},{0,1},{0,0},{0,1}}

-- PARAMS
	-- Structure (inputs, [hiddens], output)
	STRUCTURE 		= {2,10,2}
	learning_rate 	= 0.8
	iterations 		= 10000
	epochs 			= 10


-- START PROCESS --
-- Create synapse matrix
syns = createStructure(STRUCTURE)

-- Print synapses
if(inargs("-sh")) then 
	print("synapses start value:") print_r(syns) 
end

-- INIT FOR LEARNING LOOP
it_count=0
epch=math.floor(iterations/epochs)

print("check:")
for i=1,#inp do
	local inputs = ""
	print_r(forward(inp[i]))
end

-- LEARNING LOOP
-- print("== Learning begin! ==")
hr()
for iteration=1,iterations do
	-- ONE STEP
	-- Create changes matrix of structure full of zeros
	changes_matrix = createStructure(STRUCTURE, true)

	errs=0
	for i=1,#inp do

		-- Forward pass per input/output set, return array of outputs
		out = forward(inp[i])
		errs = errs + (MSE(out,exp_out[i]))

		-- Backwards pass
		local deltas = backward(out,exp_out[i],learning_rate)
		changes_matrix = addweights(changes_matrix,deltas)

		-- Debug printouts
		if(inargs("-s")) then
			print("inputs:	",inp[i][1],inp[i][2])
			print("expected:",exp_out[i])
			print("output: ",out)
			print("Pass "..iteration, "Error: "..MSE(out,exp_out[i]))
			print_r(deltas)
		end
	end
	it_count=it_count+1
	if(it_count%epch==0 or it_count==1) then
		print("Average Error over dataset: "..errs/#inp.."\r")
	end
	syns = addweights(syns,changes_matrix)
end


-- final results
hr()
if(inargs("-sh")) then
	print("synapses end value")
	print_r(addweights(syns,changes_matrix))
end



print("check:")
for i=1,#inp do
	local inputs = ""
	print_r(forward(inp[i]))
end

-- print("check:")
-- for i=1,#inp do
-- 	local inputs = ""
-- 	for j=1,#inp[i] do
-- 		if(inputs~="") then
-- 			inputs = inputs..","
-- 		end
-- 		inputs=inputs..inp[i][j]
-- 	end

-- 	-- local status = ""
-- 	print(inputs.." -> "..exp_out[i][1]..":",forward(inp[i])[1].." -> "..round(forward(inp[i])[1]))
-- end
