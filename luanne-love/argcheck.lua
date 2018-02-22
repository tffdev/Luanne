-- False argument checking
possible_args = {"-sh","-h","-s"}
for i=1,#arg do
	local inpossibleargs=false;
	for k=1,#possible_args do
		if(arg[i]==possible_args[k]) then
			inpossibleargs=true
		end
	end
	if(not inpossibleargs) then
		print("\n################### LUANNE #####################")
		print(" ERROR: ARGUMENT "..arg[i].." NOT RECOGNISED.")
		print(" Type argument '-h' for help")
		print("################################################\n")
		die()
		-- print(getcontent("help")) die()
	end
end
-- Help
if(inargs("-h")) then 
	print(getcontent("help")) 
	die() 
end