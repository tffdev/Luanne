-- LOGIC
function forward(input)

	-- CHECK THIS FUNCTION WITH NEW SYNAPSE STRUCTURE
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
	final_output = layers[#layers]
	return final_output
end

function backward(output, expected_output, ln_rate)
	deltas={}

	-- CALCULATE FIRST LAYER DELTAS
	-- init array's group-dimensions
	deltas[#syns] = {}
	gamma[#syns] = {}
	-- syns[layer][going_to][coming_from]
	-- For the length of the final layer
	for p_to=1,#syns[#syns] do 
		-- init array X-dimensions
		deltas[#syns][p_to]={}
		gamma[#syns][p_to]={}
		-- For every output neuron
		for p_from=1,#syns[#syns][p_to] do
			-- deltas[weight layer][going_to][coming_from]
			gamma[#syns][p_to][p_from] = (output[p_to]-expected_output[p_to]) * dsig(output[p_to])
			deltas[#syns][p_to][p_from] = -gamma[#syns][p_to][p_from] * (layers[#layers-1][p_from]) * ln_rate
		end
	end

	-- Calculate hidden layer deltas

	
	-- Summate gamma values for layer in front per neuron
	-- summations = {}

	-- for every layer of synapses (exc. output)
	for layer = #syns-1,1, -1 do
		local inp_str = ""
		-- for i=1,#expected_output do
		-- 	inp_str=inp_str.." "..expected_output[i]
		-- end
		-- print("altering layer "..layer.." on input "..inp_str)


		gamma[layer] = {}
		deltas[layer] = {}

		for p_to=1,#syns[layer] do
			gamma[layer][p_to] = {}
			deltas[layer][p_to] = {}
			for p_from=1,#syns[layer][p_to] do

				local summation = 0
				for i=1,#layers[layer+2] do
					summation = summation + gamma[layer+1][i][p_from]
				end
				-- print("gamma:")
				-- print_r(gamma)
				-- print("summation for ".."H"..p_to..p_from..": "..summation)
				-- for every weight
				gamma[layer][p_to][p_from] = summation * dsig(layers[layer+1][p_from]) 

				deltas[layer][p_to][p_from] = gamma[layer][p_to][p_from] * layers[layer][p_from] * ln_rate
				-- print(layers[layer+1][p_from]..", "..p_from,"gamma: "..gamma[layer][p_to][p_from])
			end
		end
	end


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

function createStructure(struct,zeros)
	zeros = false or zeros
	local newstruct = {}
	if(zeros) then
		for w=1,#struct-1 do
			newstruct[w] = m.zeros(struct[w],struct[w+1])
		end
		return newstruct
	else
		for w=1,#struct-1 do
			newstruct[w] = m.random(struct[w],struct[w+1])
		end
		return newstruct
	end
end