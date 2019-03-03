--[[

Micro Neural Net Framework [For Project 'NLPS', Daniel Brier]
--------------------------------
Synapse indexing is as follows:
	syns[weight layer][going_to][coming_from]

]]--

local luanne = require("./lib/luanne")

-- Declarations/inits#
-- TO MOVE ALL THIS INTO AN OBJECT

function main()
	local steppy = 100
	local nn = luanne:new_network( { 2, 2, 1 } )

	while steppy > 0 do
		steppy = steppy - 1
		luanne:learn( { 0, 1 }, { 1 } )
	end

end

main()