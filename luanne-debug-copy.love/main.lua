function love.load()
	love.window.setMode(1366, 768, {resizable=true, vsync=true})

	reqs = {"json","funcs","nnfuncs"}
	for i=1,#reqs do require(reqs[i]) end

	-- Declarations
	syns = {}
	layers = {}
	gamma = {}
	math.randomseed(123)

	inp = {{0,0},{0,1},{1,1},{1,0}}
	-- exp_out = {{0,0},{0,1},{0,0},{0,1}}
	exp_out = {{0},{1},{0},{1}}

	STRUCTURE 		= {2,2,1}
	learning_rate 	= 1
	iterations_per_frame = 1
	updates_per_sec = 5
	epochs 			= 10
	syns = createStructure(STRUCTURE)
	syns = {
		{
			{0.51,-0.51},{-0.51,0.51}
		},{
			{0.51,0.51}
		}
	}
	inp_count = 1
	node_table = {}
	iter_count = 1
	mouse_has_selection = false
	
	resetnodes()

	-- first pass
	fullpass()
end

function love.update()
	-- Actual loop
	if(iter_count%(math.ceil(60/updates_per_sec))==0) then
		for i=1,iterations_per_frame do
			fullpass()
		end
	end
	iter_count = iter_count+1
end

function fullpass()
	changes_matrix = createStructure(STRUCTURE, true)
	errs=0
	-- inp_count = inp_count + 1
	-- if(inp_count > #inp ) then inp_count = 1 end
	for i=1,#inp do
		out = forward(inp[i])
		errs = errs + MSE(out,exp_out[i]);
		local deltas = backward(out,exp_out[i],learning_rate)
		changes_matrix = addweights(changes_matrix,deltas)
	end
	syns = subweights(syns,changes_matrix)
end

function love.draw()
	-- Node dragging
	local mousex, mousey = love.mouse.getPosition()
	for i=1,#node_table do
		for j=1,#node_table[i] do
			if( mouse_has_selection==false and love.mouse.isDown(1) and math.pow(node_table[i][j].x-mousex,2)+math.pow(node_table[i][j].y-mousey,2) <= 800) then
				node_table[i][j].selected = true
				mouse_has_selection = true
			end
			if( love.mouse.isDown(1) and node_table[i][j].selected == true) then
				node_table[i][j].x = mousex
				node_table[i][j].y = mousey 
			end
			if(not love.mouse.isDown(1)) then
				node_table[i][j].selected = false
				mouse_has_selection = false
			end
		end
	end

	-- DRAW
	love.graphics.print("ERROR: "..errs/#inp,5,5)

	-- Reset nodes button
	love.graphics.setColor(150, 50, 50, 255)
	love.graphics.rectangle("fill",895,5,90,25)
	love.graphics.setColor(255, 255, 255, 255)
	love.graphics.print("Reset Nodes",900,10)
	if(love.mouse.isDown(1) and mousex > 895 and mousex < 895+90 and mousey > 5 and mousey < 30) then
		resetnodes()
	end

	-- draw lines
	for x=1,#node_table-1 do
		for y_from=1,#node_table[x] do
			for y_to=1,#node_table[x+1] do
				love.graphics.setLineWidth(math.sqrt(math.sqrt(math.abs(syns[x][y_to][y_from]))))
				love.graphics.setColor(-255*syns[x][y_to][y_from], 255*syns[x][y_to][y_from], 20, 255)
				love.graphics.line(node_table[x][y_from].x,node_table[x][y_from].y,node_table[x+1][y_to].x,node_table[x+1][y_to].y)
				love.graphics.setLineWidth(1)
				-- print weight
				love.graphics.printf("syns"..y_to..","..y_from..":"..tostring(round2(syns[x][y_to][y_from],3)), node_table[x][y_from].x+(node_table[x+1][y_to].x-node_table[x][y_from].x)/2 - 50, node_table[x][y_from].y+(node_table[x+1][y_to].y-node_table[x][y_from].y)/2,100,"center")
			end
		end
	end

	-- draw nodes
	for x=1,#node_table do
		for y=1,#node_table[x] do
			love.graphics.setColor(0, 0, 0, 255)
			love.graphics.circle("fill", node_table[x][y].x, node_table[x][y].y, 10)
			love.graphics.setColor(255, 255, 255, 255)
			love.graphics.circle("fill", node_table[x][y].x, node_table[x][y].y, 10*layers[x][y])
			love.graphics.setColor(255, 255, 255, 255)
			love.graphics.circle("line", node_table[x][y].x, node_table[x][y].y, 10)
		end
	end

	love.graphics.setColor(255, 255, 255, 255)

	-- for dellen=1,#deltas do
	-- 	love.graphics.print("Deltas"..dellen,(dellen-1)*300+10,40)
	-- 	for i=1,#deltas[dellen] do
	-- 		for j=1,#deltas[dellen][i] do
	-- 			love.graphics.print(i..","..j..": "..round2(deltas[dellen][i][j],3), (dellen-1)*300+(i-1)*70+10, j*30+40)
	-- 		end
	-- 	end
	-- end
	
	for x=1,#node_table do
		for y=1,#node_table[x] do
			love.graphics.print(layers[x][y], node_table[x][y].x-10, node_table[x][y].y-30)
			local extra = "Hidden "
			if(x==1) then extra = "Input " end
			if(x==#node_table) then extra = "Output " end
			love.graphics.print(extra..x..", "..y, node_table[x][y].x-10, node_table[x][y].y-45)
		end
	end
end