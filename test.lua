local luanne = require("luanne")
local nn = luanne:new_network( { 2, 3, 1 } , 0.1, 0.03)
	
-- Do 20 * 10,000 iterations, printing the average error every 10,000!
for _ = 1, 20 do
	local err = 0
	for _ = 1, 10000 do
		err = err + nn:learn( { 0, 0 }, { 0 } )
		err = err + nn:learn( { 0, 1 }, { 1 } )
		err = err + nn:learn( { 1, 1 }, { 0 } )
		err = err + nn:learn( { 1, 0 }, { 1 } )
	end
	err = err / (20*1000)
	print(string.format("Average error: %0.5f", err))
end

print(string.format(
[[--------------------
Results of XOR Test!
{0,0} = %.1f	Expected ~(0.0)
{0,1} = %.1f	Expected ~(1.0)
{1,0} = %.1f	Expected ~(1.0)
{1,1} = %.1f	Expected ~(0.0)]],
nn:forward({0,0})[1],
nn:forward({0,1})[1],
nn:forward({1,0})[1],
nn:forward({1,1})[1])
)
