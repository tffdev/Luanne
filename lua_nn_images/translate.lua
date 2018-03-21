
require("tablesave")
json = require("json")

for step=0,10000 do
	local file = io.open("ml_json_cats/"..step,"rw")
	local result = json.decode(tostring(file:read("*a")))
	table.save(result,"ml_json_cats/"..step)
	file:close()
end
