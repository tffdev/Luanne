
-- MATRIX LIBRARY, FOR MATRICES
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

return matrix_utilities