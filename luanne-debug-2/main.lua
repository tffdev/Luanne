function love.load()
	-- Declarations
	syns = {}
	nodes = {}
	gamma = {}
	inp_count = 1
	node_table = {}
	iter_count = 1
	mouse_has_selection = false
	show_values = true
	mousepress={}

	reqs = {"json","funcs","nnfuncs","config"}
	for i=1,#reqs do require(reqs[i]) end

	syns = createStructure(STRUCTURE)
	-- syns = {{{0.3,-0.3},{-0.3,0.3}},{{0.3,0.3}}}
	mouseposprev = {x=0,y=0}
	outputs = {}
	resetnodes()
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
		outputs[i] = out
		errs = errs + MSE(out,exp_out[i]);
		backward_output(out,exp_out[i],learning_rate)

		for lyr=#syns-1,1,-1 do
			backward_hidden(lyr,learning_rate)
		end

		changes_matrix = addweights(changes_matrix,deltas)
	end
	syns = subweights(syns,changes_matrix)
end


function love.draw()
	-- Node dragging
	local mousex, mousey = love.mouse.getPosition()

	if love.mouse.isDown(3) then
		for i=1,#node_table do
       		for j=1,#node_table[i] do
       			node_table[i][j].x = node_table[i][j].x + (mousex-mouseposprev.x)
       			node_table[i][j].y = node_table[i][j].y + (mousey-mouseposprev.y)
       		end
       	end
	end

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
	local err = errs/#inp
	love.graphics.setColor(255*(err*255), 255/(err*255), 0, 255)
	love.graphics.print("ERROR: "..err,15,10)
	love.graphics.setColor(255, 255, 255, 255)

	for i=1,#outputs do
		for j=1,#outputs[i] do
			love.graphics.print("Output"..i.."("..exp_out[i][j].."): "..round2(outputs[i][j],6),40+(j-1)*190,i*30)
		end
		
	end

	-- Reset nodes button
	if (draw_button(400,5,110,25,"Reset Nodes")) then
		resetnodes()
		-- Actually reset values
		-- syns = createStructure(STRUCTURE)
	end
	if (draw_button(400,35,110,25,"Turn off values")) then
		show_values = false
	end
	if (draw_button(400,65,110,25,"Turn on values")) then
		show_values = true
	end
	if(draw_button(400,95,110,25,"Reset Weights"))then
		syns = createStructure(STRUCTURE)
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
				if(show_values) then
					love.graphics.printf("syns"..y_to..","..y_from..":"..tostring(round2(syns[x][y_to][y_from],6)), node_table[x][y_from].x+(node_table[x+1][y_to].x-node_table[x][y_from].x)/2 - 50, node_table[x][y_from].y+(node_table[x+1][y_to].y-node_table[x][y_from].y)/2,100,"center")
				end
			end
		end
	end

	-- draw nodes
	for x=1,#node_table do
		for y=1,#node_table[x] do
			love.graphics.setColor(0, 0, 0, 255)
			love.graphics.circle("fill", node_table[x][y].x, node_table[x][y].y, 10)
			love.graphics.setColor(255, 255, 255, 255)
			love.graphics.circle("fill", node_table[x][y].x, node_table[x][y].y, 10*nodes[x][y])
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
	if(show_values) then
		for x=1,#node_table do
			for y=1,#node_table[x] do
				love.graphics.print(nodes[x][y], node_table[x][y].x-10, node_table[x][y].y-30)
				local extra = "Hidden "
				if(x==1) then extra = "Input " end
				if(x==#node_table) then extra = "Output " end
				love.graphics.print(extra..x..", "..y, node_table[x][y].x-10, node_table[x][y].y-45)
			end
		end
	end
	mouseposprev.x = mousex
	mouseposprev.y = mousey
	mousepress={}
end


-- BUTTONS
function draw_button(x,y,w,h,text)
	local mousex, mousey = mousepress.x, mousepress.y
	love.graphics.setColor(150, 50, 50, 255)
	love.graphics.rectangle("fill",x,y,w,h)
	love.graphics.setColor(255, 255, 255, 255)
	love.graphics.printf(text,x,y+math.floor(h/2)-6,w,"center")
	if(mousepress.button==1 and mousex > x and mousex < x+w and mousey > y and mousey < y+h) then
		return true	
	else
		return false
	end
end
function love.mousepressed( x, y, button )
	mousepress={}
	mousepress.x = x
	mousepress.y = y
	mousepress.button = button
end
function love.wheelmoved(x, y)
	local mousex, mousey = love.mouse.getPosition()

    if y > 0 then
       	for i=1,#node_table do
       		for j=1,#node_table[i] do
       			node_table[i][j].x = node_table[i][j].x + (node_table[i][j].x - mousex)/5
       			node_table[i][j].y = node_table[i][j].y + (node_table[i][j].y - mousey)/5
       		end
       	end
    end
    if y < 0 then
       	for i=1,#node_table do
       		for j=1,#node_table[i] do
       			node_table[i][j].x = node_table[i][j].x + (mousex-node_table[i][j].x)/5
       			node_table[i][j].y = node_table[i][j].y + (mousey-node_table[i][j].y )/5
       		end
       	end
    end
end