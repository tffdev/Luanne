# Luanne
An ultra-minimal neural network library. If you can vectorise your data, you can make machines learn anything!

## Dependencies:
* luafilesystem `sudo luarocks install luafilesystem`

## Basic Usage

```lua
local luanne = require("./lib/luanne")

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