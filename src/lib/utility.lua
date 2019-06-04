
-- UTILITY LIBRARY
-- ====================
-- Contains things commonlu used debugging tools or anything 
-- we'd use for our interface.

function printf(...)
    print(string.format(...))
end

function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function round(x)
	if(x-math.floor(x)<0.5) then
		return math.floor(x)
	else
		return math.ceil(x)
	end
end

function inargs(tosearch)
	local h = false
	for i=1,#arg do
		if(arg[i]==tosearch) then h = true end
	end
	return h
end

function fs_save_weights(name,array)
	local file = io.open(name, "w")
	file:write(json.encode(array))
	file:close()
end

function fs_load_weights(name)
	local file = io.open(name, "r")
	local json = json.decode(file:read())
	file:close()
	return json
end	

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

function die()
  os.exit(1)
end

function read_file(file)
	local f = io.open(file, "r")
    local content = f:read("*all")
    f:close()
    return content
end

function file_exists(name)
   local f = io.open(name,"r")
   if f ~= nil then io.close(f) return true else return false end
end
