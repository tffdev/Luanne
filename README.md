# Luanne
An ultra-minimal neural network library. Function approximation for vectorised data!

## Disclosure
This library is slower than more sophisticated solutions due to not having integration
with parallel-processing hardware, however it is fast enough for simple use cases.

## Basic Usage

`local my_network = luanne:new_network(STRUCTURE, LEARNING RATE, MOMENTUM)` 
- **STRUCTURE**: a table that represents your network's structure. Each number = how many nodes that layer contains. FIRST ENTRY MUST EQUAL INPUT SIZE, LAST ENTRY MUST EQUAL OUTPUT SIZE. E.g. XOR takes 2 inputs, 1 output. Structure must be {2, ..., 1}
- **LEARNING RATE**: a small number, usually 0.01. This is how far each learning iteration "kicks" the network's weights.
- **MOMENTUM**: a small number between 0.0 and 1.0. This is how much the previous iteration influences the next iteration.

`my_network:learn(INPUT, OUTPUT)`
- Where **INPUT** and **OUTPUT** are your networks input and output vectors.
- **RETURNS** the mean squared error for that iteration. Used for seeing how well your network is doing. Smaller = better

`local result = my_network:forward(INPUT)`
Perform a normal forward pass on your final network.

```lua
local luanne = require("luanne")

-- Learning XOR 

function main()
	local nn = luanne:new_network( { 2, 3, 1 } )
	for _ = 1, 1000 do
		nn:learn( { 0, 0 }, { 0 } )
		nn:learn( { 0, 1 }, { 1 } )
		nn:learn( { 1, 1 }, { 0 } )
		nn:learn( { 1, 0 }, { 1 } )
	end

	-- Test
	print("Output for {1,0}: " .. nn:forward({ 1, 0 })[1])
end

main()
```

## SAVING YOUR NETWORK
Use one of the many libraries for Lua that lets you save tables. e.g. http://lua-users.org/wiki/SaveTableToFile
```lua
-- Saving your network
local my_network = luanne:new_network({2,3,1})
-- *LEARN HERE*
table.save(my_network.synapses, "my_network.txt")

-- Loading your network. Make sure the structure is the same!
local my_network = luanne:new_network({2,3,1})
my_network.synapses = table.load("my_network.txt")
```
