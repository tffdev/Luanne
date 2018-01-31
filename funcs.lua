function sig(x)
	return 1/(1+math.exp(-x))
end

function dsig(x)
	return sig(x)*(1-sig(x))
end

function relu(x)
	if x < 0 then return 0 else return x end
end

function round(x)
	if(x-math.floor(x)<0.5) then
		return math.floor(x)
	else
		return math.ceil(x)
	end
end

function MSE(input,expected)
	local total = 0
	for i=1,#input do
		total = total + (math.pow(input[i]-expected[i],2)/2)
	end
	return total
end

function inargs(tosearch)
	local h = false
	for i=1,#arg do
		if(arg[i]==tosearch) then h = true end
	end
	return h
end

weights = {
	save = function(name,array)
		local file = io.open(name, "w")
		file:write(json.encode(array))
		file:close()
	end,
	load = function(name)
		local file = io.open(name, "r")
		local json = json.decode(file:read())
		file:close()
		return json
	end	
}

-- MATRIX LIBRARY, FOR MATRICES
m = {
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
		for i=1,#inp1 do total = total + (inp1[i]*inp2[i]) end
		return total
	end,

	random = function(width,height)
		local final = {}
			for i=1,height do
				local temp = {}
				for j=1,width do
					temp[j] = math.random(-1000000,1000000)/1000000
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


function print_r(arr, indentLevel)
    local str = ""
    local indentStr = "#"

    if(indentLevel == nil) then
        print(print_r(arr, 0))
        return
    end

    for i = 0, indentLevel do
        indentStr = indentStr.."\t"
    end

    for index,value in pairs(arr) do
        if type(value) == "table" then
            str = str..indentStr..index..": \n"..print_r(value, (indentLevel + 1))
        else 
            str = str..indentStr..index..": "..value.."\n"
        end
    end
    return str
end

function hr()
	print("-----------------------------------")
end

function die ()
  os.exit(1)
end

function getcontent(file)
	local f = io.open(file, "r")
    local content = f:read("*all")
    f:close()
    return content
end