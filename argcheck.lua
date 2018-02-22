-- False argument checking
possible_args = {"-sh","-h","-s","-learn"}
for i=1,#arg do
	local inpossibleargs=false;
	for k=1,#possible_args do
		if(arg[i]==possible_args[k]) then
			inpossibleargs=true
		end
	end
	if(not inpossibleargs) then
		print(" ERROR: Argument "..arg[i].." not recognised.")
		print(" Type argument '-h' for help")
		die()
		-- print(getcontent("help")) die()
	end
end
-- Help
if(inargs("-h")) then 
	print(getcontent("help")) 
	die() 
end