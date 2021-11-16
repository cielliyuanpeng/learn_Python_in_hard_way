from sys import argv
from os.path import exists

script,from_file,to_file = argv

print("Copy from {} ,to {}".format(from_file,to_file))

in_file = open(from_file)
indata = in_file.read()

print("The input file is {} bytes long".format(len(indata)))

print("Does the out file exist?".format(exists(to_file)))

out_file = open(to_file,'w')
out_file.write(indata)

print("all done")

out_file.close()
in_file.close()
