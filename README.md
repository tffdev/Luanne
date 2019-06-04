# Luanne
An ultra-minimal neural network library. Function approximation for vectorised data!

## Disclosure
This library, in the context of many methods of machine learning, is extremely slow due to not having integration
with parallel-processing hardware. However it was fast enough to use for an image classification program (classic cat-or-dog).

## Dependencies:
* LuaJIT (For 100x the speed of Lua5.1)
* luafilesystem `sudo luarocks install luafilesystem`

## Basic Usage
Check the main.lua example file in src for more details

```lua
local luanne = require("./lib/luanne")

-- Learning XOR 

function main()
	local nn = luanne:new_network( { 2, 2, 1 } )
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