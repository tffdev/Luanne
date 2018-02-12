-- LOGIC
function forward(input)
	-- CHECK THIS FUNCTION WITH NEW SYNAPSE STRUCTURE
	final_output = {}
	layers[1] = input
	-- for each layer of synapses
	for s=1, #syns do
		layers[s+1] = {}
		for i=1,#syns[s] do
			layers[s+1][i] = sig(m.dot(layers[s],syns[s][i]))
		end
	end
	final_output = layers[#layers]
	return final_output
end

function backward(output, expected_output, ln_rate)
	deltas={}
	-- CALCULATE FIRST LAYER DELTAS
	deltas[#syns] = {}
	gammas[#layers] = {}

	for p_to=1,STRUCTURE[#STRUCTURE] do 
		deltas[#syns][p_to] = {}
		gammas[#layers][p_to] = ((2*#output)/4)*(output[p_to] - expected_output[p_to]) * dsig(output[p_to])
		-- For every output neuron
		for p_from=1,STRUCTURE[#STRUCTURE-1] do
			deltas[#syns][p_to][p_from] = - gammas[#layers][p_to] * (layers[#layers][p_to]) * ln_rate
		end
	end

	-- CALCULATE HIDDEN LAYER DELTAS
	for layer = #syns-1,1,-1 do
		-- print("doing layer "..layer)
		gammas[layer+1] = {}
		deltas[layer] = {}

		-- Calc gammass for every neuron
		for current_neuron=1,#layers[layer+1] do
			-- print("p to "..current_neuron.." layer "..layer)
			gammas[layer+1][current_neuron] = {}
			deltas[layer][current_neuron] = {}

			local summation = 0
			for i=1,#layers[layer+2] do
				summation = summation + (gammas[layer+2][i] * syns[layer+1][i][current_neuron])
			end
			gammas[layer+1][current_neuron] = summation * dsig(layers[layer+1][current_neuron])
		end

		-- Calc deltas
		for p_to=1,#syns[layer+1] do
			deltas[layer][p_to] = {}
			for p_from=1,#syns[layer][p_to] do
				deltas[layer][p_to][p_from] = gammas[layer+1][p_to] * layers[layer+1][p_from] * ln_rate
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