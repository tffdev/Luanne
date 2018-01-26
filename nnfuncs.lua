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

function backward(output, expected_output_index, expected_output, ln_rate)
	deltas={}

	-- CALCULATE FIRST LAYER DELTAS
	-- init array's group-dimensions
	deltas[#syns]={}

	for i=1,#syns[#syns] do
		-- init array X-dimensions
		deltas[#syns][i]={}

		for j=1,#syns[i] do
			-- deltas[weight layer][neuron output layer][index of weight]
			deltas[#syns][i][j] = -(output-expected_output[expected_output_index]) * sigderiv(output) * (layers[#layers-1][j]) * ln_rate
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

function sleep(n)
  os.execute("sleep " .. tonumber(n))
end