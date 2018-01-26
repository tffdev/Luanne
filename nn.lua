--[[

Micro Neural Net Framework [For NLPS, Daniel Brier]
--------------------------------
Help:
	Synapse indexing is as follows: 
		syns[weight_set][number_of_proceeding_neuron][number of synapse]
]]--

-- Requirements
reqs = {"json","funcs","nnfuncs"}
for i=1,#reqs do require(reqs[i]) end

-- Help
if(inargs("-h")) then print(getcontent("help")) die() end

--inits()
syns = {}
layers = {}
math.randomseed(123)

-- DATASET
inp = {{0,0},{0,1},{1,1},{1,0}}
exp_out = {0,1,0,1}

-- PARAMS
-- Structure (inputs, [hiddens], output)
STRUCTURE = {2,3,1}
learning_rate = 0.5

-- Create synapse matrix
for w=1,#STRUCTURE-1 do
	syns[w] = m.random(STRUCTURE[w],STRUCTURE[w+1])
end

-- Print synapses
if(inargs("-s")) then 
	print("synapses start value:") print_r(syns) 
end





-- INIT FOR LEARNING LOOP
it_count=0
print("== Learning begin! ==")
hr()


-- LEARNING ULTRALOOP
for iteration=1,100000 do
	-- ONE STEP
	changes_matrix = {}
	for w=1,#STRUCTURE-1 do
		changes_matrix[w] = m.zeros(STRUCTURE[w],STRUCTURE[w+1])
	end

	errs=0
	for i=1,#inp do

		-- Forward pass per input/output set
		out = forward(inp[i])
		errs=errs+MSE(out,exp_out[i]);
		local deltas = backward(out,i,exp_out,learning_rate)
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
	if(it_count%20000==0) then
		print("Average Error over dataset: "..errs/#inp)
	end
	it_count=it_count+1
	syns = addweights(syns,changes_matrix)
end


-- final results
hr()
if(inargs("-s")) then
	print("synapses end value")
	print_r(addweights(syns,changes_matrix))
end

print("check:")
print("0,0 -> 0:",forward(inp[1]))
print("0,1 -> 1:",forward(inp[2]))
print("1,1 -> 0:",forward(inp[3]))
print("1,0 -> 1:",forward(inp[4]))