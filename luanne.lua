-- MATRIX LIBRARY
-- ====================
-- All the matrix math we need

local matrix_utilities = {
	summate = function(array)
		local total = 0
		for i=1,#array do
			total = total + array[i]
		end
		return total
	end,
	diff = function(ar1,ar2,abs)
		local error_array = {}
		for i=1,#ar1[1] do
			error_array[i] = ar2[1][i] - ar1[1][i]
			if abs==true then error_array[i] = math.abs(error_array[i]) end
		end
		return error_array
	end,
	average = function(input_matrix)
		local final_array = 0
		for i=1,#input_matrix do
			final_array = final_array + input_matrix[i]
		end
		return final_array / #input_matrix
	end,

	dot = function(inp1,inp2)
		local total = 0
		if(#inp1~=#inp2) then return false end
		for i=1,#inp1 do 
			total = total + (inp1[i]*inp2[i])
		end
		return total
	end,

	random = function(width,height)
		local final = {}
			for i=1,height do
				local temp = {}
				for j=1,width do
					temp[j] = math.random()-0.5
				end
				table.insert(final, temp)
			end
		return final
	end,
	zeros = function(width,height)
		local final = {}
			for i=1,height do
				local temp = {}
				for j=1,width do
					temp[j] = 0
				end
				table.insert(final, temp)
			end
		return final
	end,

	transpose = function(input_matrix)
		local output = {}
		for i=1, #input_matrix do
			for j=1, #input_matrix[i] do
				output[j] = output[j] or {}
				output[j][i] = input_matrix[i][j]
			end
		end
		return output
	end,

	multiply = function(matrix1,matrix2)
		--init and check
		local output = {}
		if m.len(matrix1) ~= m.height(matrix2) then return false end

		for i=1,m.height(matrix1) do 
			for j=1, m.len(matrix2) do
				output[i] = output[i] or {}
				local final = 0

				if m.len(matrix1) == 1 then 
					for c=1, m.len(matrix1) do
						final = final + (matrix1[c][1] * matrix2[i][c])
					end
				elseif m.len(matrix2) == 1 then
					for c=1, m.len(matrix2) do
						final = final + (matrix1[c][j] * matrix2[1][c])
					end
				else
					for c=1, m.len(matrix1) do
						final = final + (matrix1[c][j] * matrix2[i][c])
					end
				end

				output[i][j] = final
			end
		end
		return output
	end,

	height = function(x)
		if type(x) == "table" then
			return #x
		else
			return 1
		end
	end,

	len = function(x)
		if type(x[1]) == "table" then
			return #x[1]
		else 
			return 1
		end
	end,

	cost = function(input, expected_output)
		local output = 0
		for i=1,#input do
			output = output + math.pow(input[i]-expected_output[i],2)
		end
		return output/2
	end
}


-- LUANNE 
-- ====================
-- All the neural network logic!
local luanne = {}


-- Create a new network structure
-- NOTE synapses are randomised between 0 and 1.
-- Nodes do not need to be initialised.
function luanne:new_network(structure, rate, momentum)
	local nn = {}
	setmetatable(nn, self)

	-- set values for new network
	self.__index 	= self
	self.structure  = structure
	self.synapses 	= luanne:create_synapse_structure(structure)
	self.learning_rate = rate or 0.2
	self.momentum_multiplier = momentum or 0

	-- nil initialisations
	self.gamma 		= {}
	self.output 	= {}
	self.input 		= {}
	self.exp_out 	= {}
	self.nodes 		= {}

	return nn
end

-- Sigmoid/threshold functions
-- Implementing Sigmoid functions into our nodes introduces
-- non-linearity for our function approximation. 
-- (Our approximated function can be a crazy wobbly line 
-- rather than a straight knife-cut through an n-dimensional dataset)
function luanne:sigmoid(x)
	return 1/(1 + math.exp(-1 * x))
end

function luanne:derivative_sigmoid(x)
	return self:sigmoid(x)*(1-self:sigmoid(x))
end

function luanne:inverse_sig(x)
	return -math.log((1/x) - 1)
end

-- *Very* basic threshold function that we aren't using
-- but want to include just to be polite
function luanne:relu(x)
	if x < 0 then return 0 else return x end
end

-- Mean squared error algorithm
-- Takes two vectors, calculates the average squared difference between each adjacent value
function luanne:MSE(actual_value, expected_value)
	local total = 0
	for i = 1, #actual_value do
		total = total + (math.pow(actual_value[i] - expected_value[i], 2)/2)
	end
	return total
end

-- Forward propagation
-- Takes a vector input, performs a pass on the network and returns the output layer
function luanne:forward(input)
	-- set first node layer values to our vector input
	self.nodes[1] = input
	-- for each layer of synapses
	for s = 1, #self.synapses do
		self.nodes[s+1] = {}
		for i = 1, #self.synapses[s] do
			self.nodes[s+1][i] = luanne:sigmoid( 
				matrix_utilities.dot(self.nodes[s], self.synapses[s][i]) )
		end
	end
	return self.nodes[#self.nodes]
end

-- Backpropagate the final synapse layer (output layer) of the network.
-- This layer should always be calculated first.
function luanne:backpropagate_output_layer(actual_output, expected_output, ln_rate)
	local err_num = {}
	self.deltas = self.deltas or {}
	self.deltas[#self.synapses] = self.deltas[#self.synapses] or {}
	self.gamma[#self.nodes] = {}

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

-- Backpropagate the hidden synapse layers of the network.
-- Should be calculated AFTER output layer backpropagation
function luanne:backpropagate_hidden_layers(L, ln_rate)
	-- current layer
	self.gamma[L+1] = {}
	self.deltas[L] = self.deltas[L] or {}

	for i = 1, #self.nodes[L+1] do
		self.gamma[L+1][i] = 0
		
		-- self.gamma forward 
		for j = 1, #self.gamma[L+2] do
			self.gamma[L+1][i] = 
				self.gamma[L+1][i] 
				+ (self.gamma[L+2][j] * self.synapses[L+1][j][i])
		end

		-- As nodes contain their values post-sigmoid, we have to 
		-- invert the sigmoid function before deriving it
		self.gamma[L+1][i] = 
			self.gamma[L+1][i] * self:derivative_sigmoid(self:inverse_sig(self.nodes[L+1][i]))
	end

	-- update self.deltas
	for i=1,#self.nodes[L+1] do 
		self.deltas[L][i] = self.deltas[L][i] or {}
		
		-- for every synapse
		for j=1,#self.nodes[L] do
			local previous_weights_for_momentum = self.deltas[L][i][j] or 0
			self.deltas[L][i][j] = 
				self.gamma[L+1][i] * (self.nodes[L][j]) 
				+ previous_weights_for_momentum * self.momentum_multiplier
		end
	end

	return self.deltas
end

-- Addition of 2 three-dimensional matrices
function luanne:add_weights(m1, m2)
	local output = {}
	for i = 1, #m1 do
		output[i] = {}
		for j = 1, #m1[i] do
			output[i][j] = {}
			for k = 1, #m1[i][j] do
				output[i][j][k] = m1[i][j][k] + m2[i][j][k]
			end
		end
	end
	return output
end


-- Subtraction of 2 three-dimensional matrices
function luanne:subtract_weights(m1, m2)
	local output = {}
	for i = 1, #m1 do
		output[i] = {}
		for j = 1, #m1[i] do
			output[i][j] = {}
			for k = 1, #m1[i][j] do
				output[i][j][k] = m1[i][j][k]-m2[i][j][k] * self.learning_rate
			end
		end
	end
	return output
end

-- Given a structure (e.g. {2,2,1}), will create 
-- an array with an appropriate number of layers and depth
-- that we can use as a holder for synapse weights.
-- (fill_with_zeros should usually NOT be used)
function luanne:create_synapse_structure(struct)
	local newstruct = {}
	for w = 1, #struct - 1 do
		newstruct[w] = matrix_utilities.random(struct[w], struct[w+1])
	end
	return newstruct
end

-- Let our network perform a single iteration on
-- a dataset input and output.
-- The more often this is called with different data, the more 
-- accurate the network will eventually be.
-- RETURNS the mean squared error for this iteration, can be used for analysis
function luanne:learn(input, expected_output)
	local real_output = self:forward(input)

	-- backpropagate the final layer (output layer)
	self:backpropagate_output_layer(real_output, expected_output, self.learning_rate)

	-- backpropagate for all hidden layers
	for layer = #self.synapses-1, 1, -1 do
		self:backpropagate_hidden_layers(layer, self.learning_rate)
	end

	self.synapses = self:subtract_weights(self.synapses, self.deltas)
	return self:MSE(real_output, expected_output)
end

return luanne
