#!/usr/local/bin/lua

lfs = require "lfs"
local Bitmap = require("deps/bitmap")
require("deps/tablesave")

-- Main script
function main()
	local configuration = {}

	-- Ask user for init values
	-- {config-hash, prompt, defualt}
	local placeholderTodo = {
		{"project-name", "Name the new savestate (newproject): "},
		{"learning-rate", "What is the learning rate? (0.033): ", "0.033"},
		{"momentum", "What is your momentum multiplier? (0.3): ", "0.3"},
		{"save-per-count", "How often would you like to save your network (per # iterations)? (2500): ", "2500"},
	}
	for i=1,#placeholderTodo do
		local output = ""
		while(output == "") do
			io.write(placeholderTodo[i][2])
			output = io.read()
			if(output == "") then
				if(placeholderTodo[i][3] == nil) then
					print("You need to insert a value")
				else
					output = placeholderTodo[i][3]
				end
			end
		end
		configuration[placeholderTodo[i][1]] = output
	end
	
	-- Set step count
	configuration["images"] = ask_for_filepaths()
	save_configuration_file(configuration)

	generate_images(configuration)

	print("Init complete!")
end

function save_configuration_file(configuration)
	-- Create project folder
	lfs.mkdir("../savestates/"..configuration["project-name"])
	table.save(configuration, "../savestates/"..configuration["project-name"].."/config")
	print("Configuration file saved.\n")
end

function ask_for_filepaths()
	-- Image folder insertion
	local user_input = "-"
	local count = 1
	local output_paths_table = {}
	print("Filepaths for images you want to learn (enter blank string to finish)")
	while (1) do
		io.write("Insert filepath for image folder "..count..":")
		local filepath = io.read()
		if(filepath == "") then break; end
		io.write("Insert the category name for image folder "..count.." (e.g. Cat):")
		local category = io.read()
		if(category == "") then break; end
		output_paths_table[count] = {filepath, category}
		count = count + 1
	end
	return output_paths_table
end

function generate_images(config) 
	print("generating image vectors...")
	for i=1,#config["images"] do
		local count = 1
		print("generating folder: "..config["images"][i][2])
		lfs.mkdir("../savestates/"..config["project-name"].."/"..config["images"][i][2])
		for file in lfs.dir(config["images"][i][1]) do
			-- Process every file, save to vector
			if(string.find(file,".bmp")) then
				local img = Bitmap.from_file(config["images"][i][1].."/"..file)
				if(img) then
					local output_vector = {}
					local fail = false
					for i=1,30 do
						for j=1,30 do
							local pixel = {img:get_pixel(i,j)}
							if(pixel[1] and pixel[2] and pixel[3]) then
								table.insert(output_vector,((pixel[1]+pixel[2]+pixel[3])/3)/255)
							else
								fail = true
							end
						end
					end
					if(not fail) then
						table.save(output_vector,"../savestates/"..config["project-name"].."/"..config["images"][i][2].."/"..count)
						count = count + 1
					end
				end
			end
		end
	end
	print("Image vectors generated!")
end

main()
