--[[

Micro Neural Net Framework [For Project 'NLPS', Daniel Brier]
--------------------------------
Synapse indexing is as follows:
	syns[weight layer][going_to][coming_from]

]]--

-- Requirements
reqs = {"json","funcs","nnfuncs","argcheck","config","tablesave"}
for i=1,#reqs do require(reqs[i]) end

-- Declarations/inits
syns 		= {}
nodes 		= {}
gamma 		= {}
outputs 	= {}
it_count	= 0
epch 		= math.floor(iterations/epochs)
syns 		= createStructure(STRUCTURE)
inp 		= {{}}
exp_out 	= {{}}
config_data = {step=0}
poems 		= getcontent("poems.txt")
poems_len 	= string.len(poems)
avg_error 	= 0
-- Load previous weights
if(file_exists("synapses")) then
	syns = table.load("synapses")
	config_data = table.load("data")
end

if(inargs("-learn"))then
	while(true) do
		-- Generate data to pass through
		config_data.step = config_data.step+1
		if(config_data.step > poems_len) then 
			config_data.step = 0
		end
		gen_alphahot(
			string.lower(string.sub(poems,config_data.step,config_data.step)),
			string.lower(string.sub(poems,config_data.step+1,config_data.step+1))
		)

		changes_matrix = createStructure(STRUCTURE, true)
		fullpass()
		it_count = it_count + 1
		avg_error = avg_error + errs/#inp
		if(it_count==save_per) then
			print("Average Error over dataset: "..avg_error/save_per,"(Weights saved)")
			table.save(syns,"synapses")
			table.save(config_data,"data")
			avg_error = 0
			it_count = 0
		end
		syns = subweights(syns,changes_matrix)
	end
else
	letter = "a"
	io.write(letter)
	for i = 1, 40 do
		local letter_output = forward(gen_ff_alphahot(letter))
		letter = ungen_ff_alphahot(letter_output)
		io.write(letter)
	end
	print("")
end