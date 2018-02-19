function sig(x)
	return 1/(1+math.exp(-1*x))
end

function dsig(x)
	return sig(x)*(1-sig(x))
end

function relu(x)
	if x < 0 then return 0 else return x end
end

function MSE(input,expected)
	local total = 0
	for i=1,#input do
		total = total + (math.pow(input[i]-expected[i],2)/2)
	end
	return total
end

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
	deltas = deltas or {}

	-- CALCULATE LAST LAYER DELTAS
	deltas[#syns] = deltas[#syns] or {}
	gamma[#layers] = {}
	err_num = {}
	-- For num of outputs
	for p_to=1,#syns[#syns] do 
		err_num[p_to] = output[p_to] - expected_output[p_to]
	end
	
	for p_to=1,#syns[#syns] do 	
		gamma[#layers][p_to] = err_num[p_to] * dsig(output[p_to])
	end

	for p_to=1,#syns[#syns] do 
		deltas[#syns][p_to] = deltas[#syns][p_to] or {}
		for p_from=1,#syns[#syns-1] do
			local momentum = deltas[#syns][p_to][p_from] or 0
			deltas[#syns][p_to][p_from] = gamma[#layers][p_to] * (layers[#layers][p_to]) * ln_rate + momentum*0.8
		end
	end


	-- CALCULATE HIDDEN LAYER DELTAS
	for layer = #syns-1,1,-1 do
		gamma[layer+1] = {}
		deltas[layer] = deltas[layer] or {}
		for i=1,STRUCTURE[layer+1] do
			gamma[layer+1][i] = 0
			for j=1,#gamma[layer+2] do
				gamma[layer+1][i] = gamma[layer+1][i] + gamma[layer+2][j] * syns[layer+1][j][i]
			end
			gamma[layer+1][i] = gamma[layer+1][i] * dsig(layers[layer+1][i])
		end

		for p_to=1,#syns[layer] do 
			deltas[layer][p_to] = deltas[layer][p_to] or {}
			for p_from=1,#syns[layer] do
				local momentum = deltas[layer][p_to][p_from] or 0
				deltas[layer][p_to][p_from] = gamma[layer+1][p_to] * (layers[layer+1][p_to]) * ln_rate + momentum*0.8
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

function subweights(m1,m2)
	local output={}
	for i=1,#m1 do
		output[i]={}
		for j=1,#m1[i] do
			output[i][j]={}
			for k=1,#m1[i][j] do
				if(m2[i]~=nil and m2[i][j]~=nil and m2[i][j][k]~=nil) then
					output[i][j][k]=m1[i][j][k]-m2[i][j][k] * learning_rate
				else
					output[i][j][k]=m1[i][j][k]
				end
			end
		end
	end
	return output
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

function resetnodes()
	for i=1,#STRUCTURE do
		node_table[i] = {}
		for j=1,STRUCTURE[i] do
			node_table[i][j] = {}
			node_table[i][j].x = (i-1)*(900)/(#STRUCTURE)+100
			if(STRUCTURE[i]==1) then 
				node_table[i][j].y = (j)*(420)/2+250
			else
				node_table[i][j].y = (j-1)*(420)/(STRUCTURE[i]-1)+250
			end
			node_table[i][j].selected = false
		end
	end
end