--[[

Micro Neural Net Framework [For Project 'NLPS', Daniel Brier]
--------------------------------

This file contains an example of machine learning the function
XOR. XOR is popular as it can only be performed on a network
with functional hidden layers.

What we've done here is create a new network with an input vector 
of length 2, an output vector of length 1,
and a hidden layer of length 3.

Our learning rate (the velocity of a synapse value "kick") 
is 0.1, and Momentum is 0.03, both of which are sufficient for this small demo
but usually, the values would be MUCH smaller than this.
]]--

local luanne = require("./lib/luanne")
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
	err = err / (4*1000)
	print(string.format("Average error: %0.5f", err))
end


-- nn:forward takes an input vector, performs a pass on your network
-- and returns an output vector! This is your generated function.
test_result_string = [[
--------------------
Results of XOR Test!
{0,0} = %.5f, Expected ~(0)
{0,1} = %.5f, Expected ~(1)
{1,0} = %.5f, Expected ~(1)
{1,1} = %.5f, Expected ~(0)
]]

printf(
	test_result_string,
	nn:forward({0,0})[1],
	nn:forward({0,1})[1],
	nn:forward({1,0})[1],
	nn:forward({1,1})[1]
)