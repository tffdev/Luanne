function sig(x)
	return 1/(1+math.exp(-1*x))
end

function dsig(x)
	return sig(x)*(1-sig(x))
end
function sig_inv(x)
	return -math.log((1/x) - 1)
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
	nodes[1] = input
	-- for each layer of synapses
	for s=1, #syns do
		nodes[s+1] = {}
		for i=1,#syns[s] do
			nodes[s+1][i] = sig(m.dot(nodes[s],syns[s][i]))
		end
	end
	final_output = nodes[#nodes]
	return final_output
end

function backward_output(output, expected_output, ln_rate)
	deltas = deltas or {}

	-- CALCULATE LAST LAYER DELTAS
	deltas[#syns] = deltas[#syns] or {}
	gamma[#nodes] = {}
	err_num = {}
	-- For num of outputs
	for i=1,#nodes[#nodes] do 
		err_num[i] = output[i] - expected_output[i]
	end
	
	for i=1,#nodes[#nodes] do 	
		gamma[#nodes][i] = err_num[i] * dsig(sig_inv(output[i]))
	end

	-- update deltas
	for i=1,#nodes[#nodes] do 
		deltas[#syns][i] = deltas[#syns][i] or {}
		for j=1,#nodes[#nodes-1] do
			local momentum = deltas[#syns][i][j] or 0
			deltas[#syns][i][j] = gamma[#nodes][i] * (nodes[#nodes-1][j]) + momentum*momentum_multiplier
		end
	end
	-- Everything above works, leave it
end

function backward_hidden(syn_lyr,ln_rate)
	-- CALCULATE HIDDEN LAYER DELTAS
	-- current layer
	gamma[syn_lyr+1] = {}
	deltas[syn_lyr] = deltas[syn_lyr] or {}

	for i=1,#nodes[syn_lyr+1] do
		gamma[syn_lyr+1][i] = 0
		-- gamma forward 
		for j=1,#gamma[syn_lyr+2] do
			gamma[syn_lyr+1][i] = gamma[syn_lyr+1][i] + (gamma[syn_lyr+2][j] * syns[syn_lyr+1][j][i])
		end
		gamma[syn_lyr+1][i] = gamma[syn_lyr+1][i] * dsig(sig_inv(nodes[syn_lyr+1][i]))
	end

	-- update deltas
	for i=1,#nodes[syn_lyr+1] do 
		deltas[syn_lyr][i] = deltas[syn_lyr][i] or {}
		for j=1,#nodes[syn_lyr] do
			-- for every synapse
			local momentum = deltas[syn_lyr][i][j] or 0
			deltas[syn_lyr][i][j] = gamma[syn_lyr+1][i] * (nodes[syn_lyr][j]) + momentum*momentum_multiplier
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
				node_table[i][j].y = (j)*(420)/2+200
			else
				node_table[i][j].y = (j-1)*(420)/(STRUCTURE[i]-1)+200
			end
			node_table[i][j].selected = false
		end
	end
end

function fullpass()
	changes_matrix = createStructure(STRUCTURE, true)
	errs=0
	-- inp_count = inp_count + 1
	-- if(inp_count > #inp ) then inp_count = 1 end
	for i=1,#inp do
		out = forward(inp[i])
		outputs[i] = out
		errs = errs + MSE(out,exp_out[i]);
		backward_output(out,exp_out[i],learning_rate)

		for lyr=#syns-1,1,-1 do
			backward_hidden(lyr,learning_rate)
		end

		changes_matrix = addweights(changes_matrix,deltas)
	end
	syns = subweights(syns,changes_matrix)
end

function gen_alphahot(input_letter,expected_output_letter)
	-- Generate one-hot matrix input/outputs
	for i=1,26 do
		inp[1][i] = 0
		if(al_num[i] == input_letter)then
			inp[1][i]=1
		end
	end
	for i=1,26 do
		exp_out[1][i] = 0
		if(al_num[i] == expected_output_letter)then
			exp_out[1][i]=1
		end
	end
end