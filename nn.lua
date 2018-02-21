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
inp = {{}}
exp_out = {{}}
config_data = {step=0}
poems = getcontent("poems.txt")
poems_len = string.len(poems)
-- Load previous weights
if(file_exists("synapses")) then
	syns = table.load("synapses")
	config_data = table.load("data")
end

for iteration = 1, iterations do
	-- Generate data to pass through
	config_data.step = config_data.step+1
	if(config_data.step > poems_len) then 
		config_data.step = 0
	end
	gen_alphahot(
		string.sub(poems,config_data.step,config_data.step),
		string.sub(poems,config_data.step+1,config_data.step+1)
	)

	changes_matrix = createStructure(STRUCTURE, true)
	fullpass()
	it_count = it_count + 1
	if(it_count%epch == 0 or it_count == 1) then
		print("Average Error over dataset: "..errs/#inp,"(Weights saved)")
		table.save(syns,"synapses")
		table.save(config_data,"data")
	end
	syns = subweights(syns,changes_matrix)
end
hr()

-- final results
local perc = 0
local perccount = 0
for i = 1,#outputs do
	for j = 1,#outputs[i] do
		perc = perc + math.abs(outputs[i][j]-exp_out[i][j])
		perccount = perccount + 1
	end
end
print("Accuracy: "..round2((1-perc/perccount)*100,2).."%")
hr()