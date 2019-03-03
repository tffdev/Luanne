-- Parameters
math.randomseed(123)

-- Requirements
require("../lib/funcs")
require("../deps/tablesave")

local luanne = {}

function luanne:new_network(structure) 
	local nn = {}
	setmetatable(nn, self)

	-- set values for new network
	self.__index 	= self
	self.syns 		= {}
	self.gamma 		= {}
	self.output 	= {}
	self.input 		= {}
	self.exp_out 	= {}
	self.learning_rate = 0.1
	self.momentum_multiplier = 0
	self.structure  = structure
	self.nodes 		= luanne:create_structure(structure, true)

	return nn
end

-- mathematics functions
function luanne:sig(x)
	return 1/(1 + math.exp(-1 * x))
end

function luanne:dsig(x)
	return sig(x)*(1-sig(x))
end

function luanne:sig_inv(x)
	return -math.log((1/x) - 1)
end

function luanne:relu(x)
	if x < 0 then return 0 else return x end
end

function luanne:MSE(actual_value, expected_value)
	local total = 0
	for i = 1, #actual_value do
		total = total + (math.pow(actual_value[i] - expected_value[i], 2)/2)
	end
	return total
end

-- LOGIC
function luanne:forward(input)
	-- CHECK THIS FUNCTION WITH NEW SYNAPSE STRUCTURE
	local final_output = {}
	self.nodes[1] = input

	-- for each layer of synapses
	for s = 1, #self.syns do
		self.nodes[s+1] = {}
		for i = 1, #self.syns[s] do
			self.nodes[s+1][i] = sig( m.dot(self.nodes[s], self.syns[s][i]) )
		end
	end

	final_output = self.nodes[#self.nodes]
	return final_output
end

function luanne:backward_output(actual_output, expected_output, ln_rate)
	self.deltas = self.deltas or {}

	-- CALCULATE LAST LAYER self.DELTAS
	self.deltas[#self.syns] = self.deltas[#self.syns] or {}
	self.gamma[#self.nodes] = {}
	err_num = {}
	-- For num of outputs
	for i = 1, #self.nodes[#self.nodes] do 
		err_num[i] = actual_output[i] - expected_output[i]
	end
	
	for i = 1, #self.nodes[#self.nodes] do 	
		self.gamma[#self.nodes][i] = err_num[i] * self.dsig( self.sig_inv(actual_output[i]) )
	end

	-- update self.deltas
	for i = 1, #self.nodes[#self.nodes] do 
		self.deltas[#self.syns][i] = self.deltas[#self.syns][i] or {}
		for j = 1, #self.nodes[#self.nodes-1] do
			local previous_weights_for_momentum = self.deltas[#self.syns][i][j] or 0
			self.deltas[#self.syns][i][j] = self.gamma[#self.nodes][i] * (self.nodes[#self.nodes-1][j]) + previous_weights_for_momentum * self.momentum_multiplier
			-- this is extremely retarded
		end
	end
	-- Everything above works!! leave it alone jesus christ
end

function luanne:backward_hidden(syn_lyr,ln_rate)
	-- CALCULATE HIDDEN LAYER SELF.DELTAS
	-- current layer
	self.gamma[syn_lyr+1] = {}
	self.deltas[syn_lyr] = self.deltas[syn_lyr] or {}

	for i = 1, #self.nodes[syn_lyr+1] do
		self.gamma[syn_lyr+1][i] = 0
		-- self.gamma forward 
		for j = 1, #self.gamma[syn_lyr+2] do
			self.gamma[syn_lyr+1][i] = self.gamma[syn_lyr+1][i] + (self.gamma[syn_lyr+2][j] * self.syns[syn_lyr+1][j][i])
		end
		self.gamma[syn_lyr+1][i] = self.gamma[syn_lyr+1][i] *self.dsig(self.sig_inv(self.nodes[syn_lyr+1][i]))
	end

	-- update self.deltas
	for i=1,#self.nodes[syn_lyr+1] do 
		self.deltas[syn_lyr][i] = self.deltas[syn_lyr][i] or {}
		for j=1,#self.nodes[syn_lyr] do
			-- for every synapse
			local previous_weights_for_momentum = self.deltas[syn_lyr][i][j] or 0
			self.deltas[syn_lyr][i][j] = self.gamma[syn_lyr+1][i] * (self.nodes[syn_lyr][j]) + previous_weights_for_momentum * self.momentum_multiplier
		end
	end

	return self.deltas
end

-- addition of two three-dimensional matrices
function luanne:add_weights(m1, m2)
	local output = {}
	for i = 1, #m1 do
		output[i] = {}
		for j = 1, #m1[i] do
			output[i][j] = {}
			for k = 1, #m1[i][j] do
				if(m2[i] ~= nil and m2[i][j] ~= nil and m2[i][j][k] ~= nil) then
					output[i][j][k] = m1[i][j][k] + m2[i][j][k]
				else
					output[i][j][k] = m1[i][j][k]
				end
			end
		end
	end
	return output
end

function luanne:subtract_weights(m1, m2)
	local output = {}
	for i = 1, #m1 do
		output[i] = {}
		for j = 1, #m1[i] do
			output[i][j] = {}
			for k = 1, #m1[i][j] do
				if(m2[i] ~= nil and m2[i][j] ~= nil and m2[i][j][k] ~= nil) then
					output[i][j][k] = m1[i][j][k]-m2[i][j][k] * learning_rate
				else
					output[i][j][k] = m1[i][j][k]
				end
			end
		end
	end
	return output
end

function luanne:create_structure(struct, fill_with_zeros)
	fill_with_zeros = false or fill_with_zeros
	local newstruct = {}
	if(fill_with_zeros) then
		for w = 1, #struct - 1 do
			newstruct[w] = m.zeros(struct[w], struct[w+1])
		end
		return newstruct
	else
		for w=1, #struct - 1 do
			newstruct[w] = m.random(struct[w], struct[w+1])
		end
		return newstruct
	end
end

function luanne:learn(input, expected_output)
	local changes_matrix = self:create_structure(self.structure, true)

	local errs = 0

	local real_output = self:forward(input)
	local mse = self:MSE(real_output, expected_output)

	errs = errs + tonumber(mse)

	backward_output(real_output, expected_output, learning_rate)

	for layer = #self.syns-1, 1, -1 do
		backward_hidden(layer, learning_rate)
	end

	changes_matrix = add_weights(changes_matrix, self.deltas)


	self.syns = subtract_weights(self.syns, changes_matrix)

	return errs
end

return luanne

-- NLP Functions
-- function luanne:gen_alphahot(stride,input_letters)
-- 	-- Generate one-hot matrix input/outputs
-- 	local result = {}
-- 	result[1] = {} -- this is the input

-- 	for stride_current=1,stride do
-- 		for i=1+#alphabet*(stride_current-1), #alphabet*stride_current do
-- 			result[1][i] = 0
-- 			if(alphabet_chars[i - #alphabet*(stride_current-1)] == string.sub(input_letters,stride_current,stride_current) )then
-- 				result[1][i]=1
-- 			end
-- 		end
-- 	end
-- 	return result
-- end

-- function luanne:ungen_ff_alphahot(input_table)
-- 	local letter = 0
-- 	local curvalue = 0
-- 	local cur_i = 0
-- 	for i=1,#alphabet do
-- 		if(input_table[i] > curvalue) then
-- 			curvalue = input_table[i]
-- 			cur_i = i 
-- 		end
-- 	end
-- 	return alphabet_chars[cur_i]
-- end