--[[

Micro Neural Net Framework [For Project 'NLPS', Daniel Brier]
--------------------------------
Synapse indexing is as follows:
	syns[weight layer][going_to][coming_from]

]]--

local luanne = require("./lib/luanne")

local curses = require "curses"

local screen = curses.initscr()
function main()
	init_curses()
	local nn = luanne:new_network( { 2, 2, 1 } )

	for _ = 1, 300 do
		
		local err = 0
		for _ = 1, 1000 do
			err = err + nn:learn( { 0, 0 }, { 0 } )
			err = err + nn:learn( { 0, 1 }, { 1 } )
			err = err + nn:learn( { 1, 1 }, { 0 } )
			err = err + nn:learn( { 1, 0 }, { 1 } )
		end
		err = err / (4*1000)

		screen:clear()
		screen:mvaddstr(0, 1, "average error: " .. err)
		screen:mvaddstr(2, 1, "Synapses:")
		
		screen:mvaddstr(3, 1, "Layer " .. 1)
		for x = 1, #nn.synapses[1] do
			for y = 1, #nn.synapses[1][x] do
				screen:mvaddstr(4 + y, 1 + x*10, string.format("%.3f", nn.synapses[1][x][y]))
			end
		end

		screen:mvaddstr(9, 1, "Test Outputs:")
		screen:mvaddstr(10, 1, "{0,0}: "..nn:forward({0, 0})[1])
		screen:mvaddstr(11, 1, "{1,0}: "..nn:forward({1, 0})[1])
		screen:mvaddstr(12, 1, "{0,1}: "..nn:forward({0, 1})[1])
		screen:mvaddstr(13, 1, "{1,1}: "..nn:forward({1, 1})[1])
  		screen:refresh()
	end
end

function init_curses()
	curses.cbreak()
	curses.echo(false)	-- not noecho !
	curses.nl(false)	-- not nonl !
	screen:clear()
end

main()