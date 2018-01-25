--[[

Micro Neural Net Framework [For NLPS, Daniel Brier]
--------------------------------
Help:
	Synapse indexing is as follows: 
		syns[weight_set][number_of_proceeding_neuron][number of synapse]
	

]]--

-- Requirements
reqs = {
	"json",
	"funcs"
}
for i=1,#reqs do require(reqs[i]) end

--inits()
syns = {}
layers = {}
math.randomseed(123)

-- DATASET
inp = {{0,0},{0,1},{1,1},{1,0}}
exp_out = {0,1,0,1}

-- PARAMS
-- Structure (inputs, [hiddens], output)
STRUCTURE = {2,2,1}
learning_rate = 0.1


-- Create synapse matrix
for w=1,#STRUCTURE-1 do
	syns[w] = m.random(STRUCTURE[w],STRUCTURE[w+1])
end


-- LOGIC
function forward(input)
	final_output = {}
	layers[1] = input
	-- for each layer of synapses
	for s=1, #syns do
		-- init new table
		layers[s+1] = {}
		-- for every neuron's valency
		for i=1,#syns[s] do
			-- set value of neuron to dot product of previous layer and weight matrix
			layers[s+1][i] = sig(m.dot(layers[s],syns[s][i]))
		end
	end
	final_output = layers[#layers][1]
	return final_output
end

function backward(output,expoutindex,ln_rate)
	deltas={}
	-- CALCULATE FIRST LAYER DELTAS
	deltas[#syns]={}
	for i=1,#syns[#syns] do
		deltas[#syns][i]={}
		for j=1,#syns[i][#syns[i]] do
			-- deltas[weight layer][neuron output layer][index of weight]
			deltas[#syns][i][j] = -(output-exp_out[expoutindex]) * (math.exp(output)/math.pow(math.exp(output)+1,2)) * (layers[#layers-1][j]) * ln_rate
		end
	end

	-- Calculate hidden layer deltas
	-- deltas[#STRUCTURE-1]={}
	-- for i=1,STRUCTURE[#STRUCTURE-1]*STRUCTURE[#STRUCTURE] do
	-- 	for j=1,#syns[1][i] do
	-- 		deltas[#STRUCTURE-1][j] = (output-exp_out[expoutindex]) * (1-math.pow(output,2)) * (layers[#layers-1][j])
	-- 	end
	-- end
	return deltas
end

function addweights(m1,m2)
	local output={}
	for i=1,#m1 do
		output[i]={}
		for j=1,#m1[i] do
			output[i][j]={}
			for k=1,#m1[i][j] do
				if(m2[i]~=nil and m2[i][j]~=nil and m2[i][j][k]~=nil) then
					output[i][j][k]=m1[i][j][k]+m2[i][j][k]
				else
					output[i][j][k]=m1[i][j][k]
				end
			end
		end
	end
	return output
end



for iteration=1,10 do
	-- ONE STEP
	outputs_to_check={}
	changes_matrix = {}
	for w=1,#STRUCTURE-1 do
		changes_matrix[w] = m.zeros(STRUCTURE[w],STRUCTURE[w+1])
	end
	print("synapses start value:")
	print_r(syns)
	print("Pass "..iteration)
	for i=1,#inp do
		hr()
		print("inputs:	",inp[i][1],inp[i][2])
		print("expected:",exp_out[i])
		out = forward(inp[i])
		print("output: ",out)
		outputs_to_check[i]=out
		print("MSE Error:",MSE(out,exp_out[i]))
		-- print("wdeltas:")
		local deltas = backward(out,i,learning_rate)
		-- print_r(deltas)
		changes_matrix = addweights(changes_matrix,deltas)
	end
	syns = addweights(syns,changes_matrix)
end

hr()
print("changed synapses")
print_r(addweights(syns,changes_matrix))
