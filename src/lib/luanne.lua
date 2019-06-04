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
	self.structure  = structure
	self.synapses 	= luanne:create_synapse_structure(structure, true)
	self.learning_rate = 0.03
	self.momentum_multiplier = 0

	-- nil initialisations
	self.gamma 		= {}
	self.output 	= {}
	self.input 		= {}
	self.exp_out 	= {}
	self.nodes 		= {}

	return nn
end

-- Sigmoid/threshold functions
function luanne:sigmoid(x)
	return 1/(1 + math.exp(-1 * x))
end

function luanne:derivative_sigmoid(x)
	return self:sigmoid(x)*(1-self:sigmoid(x))
end

function luanne:inverse_sig(x)
	return -math.log((1/x) - 1)
end

function luanne:relu(x)
	if x < 0 then return 0 else return x end
end

-- Mean squared error algorithm
function luanne:MSE(actual_value, expected_value)
	local total = 0
	for i = 1, #actual_value do
		total = total + (math.pow(actual_value[i] - expected_value[i], 2)/2)
	end
	return total
end

function luanne:forward(input)
	-- set first node layer values to our vector input
	self.nodes[1] = input

	-- for each layer of synapses
	for s = 1, #self.synapses do
		self.nodes[s+1] = {}
		for i = 1, #self.synapses[s] do
			self.nodes[s+1][i] = luanne:sigmoid( 
				m.dot(self.nodes[s], self.synapses[s][i]) )
		end
	end
	return self.nodes[#self.nodes]
end

function luanne:backpropagate_output_layer(actual_output, expected_output, ln_rate)
	self.deltas = self.deltas or {}

	-- CALCULATE LAST LAYER self.DELTAS
	self.deltas[#self.synapses] = self.deltas[#self.synapses] or {}
	self.gamma[#self.nodes] = {}
	err_num = {}

	-- For num of outputs
	for i = 1, #self.nodes[#self.nodes] do 
		err_num[i] = actual_output[i] - expected_output[i]
	end
	
	for i = 1, #self.nodes[#self.nodes] do
		self.gamma[#self.nodes][i] = err_num[i] * self:derivative_sigmoid( self:inverse_sig(actual_output[i]) )
	end


	-- update self.deltas
	for i = 1, #self.nodes[#self.nodes] do 
		self.deltas[#self.synapses][i] = self.deltas[#self.synapses][i] or {}
		for j = 1, #self.nodes[#self.nodes-1] do
			local previous_weights_for_momentum = self.deltas[#self.synapses][i][j] or 0
			
			self.deltas[#self.synapses][i][j] = 
				self.gamma[#self.nodes][i] * (self.nodes[#self.nodes-1][j]) 
				+ previous_weights_for_momentum * self.momentum_multiplier
		end
	end
	-- Everything above works!! 
end

function luanne:backpropagate_hidden_layers(syn_lyr,ln_rate)
	-- CALCULATE HIDDEN LAYER SELF.DELTAS
	-- current layer
	self.gamma[syn_lyr+1] = {}
	self.deltas[syn_lyr] = self.deltas[syn_lyr] or {}

	for i = 1, #self.nodes[syn_lyr+1] do
		self.gamma[syn_lyr+1][i] = 0
		-- self.gamma forward 
		for j = 1, #self.gamma[syn_lyr+2] do
			self.gamma[syn_lyr+1][i] = self.gamma[syn_lyr+1][i] + (self.gamma[syn_lyr+2][j] * self.synapses[syn_lyr+1][j][i])
		end
		self.gamma[syn_lyr+1][i] = self.gamma[syn_lyr+1][i] *self:derivative_sigmoid(self:inverse_sig(self.nodes[syn_lyr+1][i]))
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
					output[i][j][k] = m1[i][j][k]-m2[i][j][k] * self.learning_rate
				else
					output[i][j][k] = m1[i][j][k]
				end
			end
		end
	end
	return output
end

function luanne:create_synapse_structure(struct, fill_with_zeros)
	fill_with_zeros = false or fill_with_zeros
	local newstruct = {}
	for w = 1, #struct - 1 do
		if fill_with_zeros then 
			newstruct[w] = m.zeros(struct[w], struct[w+1])
		else 
			newstruct[w] = m.random(struct[w], struct[w+1])
		end
	end
	-- print("newstruct:")
	-- print_r(newstruct)
	return newstruct
end

function luanne:learn(input, expected_output)
	local changes_matrix = self:create_synapse_structure(self.structure, true)

	local errs = 0

	local real_output = self:forward(input)
	local mse = self:MSE(real_output, expected_output)

	errs = errs + tonumber(mse)

	self:backpropagate_output_layer(real_output, expected_output, self.learning_rate)

	for layer = #self.synapses-1, 1, -1 do
		self:backpropagate_hidden_layers(layer, self.learning_rate)
	end

	changes_matrix = self:add_weights(changes_matrix, self.deltas)

	self.synapses = self:subtract_weights(self.synapses, changes_matrix)
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