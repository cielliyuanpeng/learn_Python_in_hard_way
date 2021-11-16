from sys import argv

script,filename = argv

print("We are going to erase{}".format(filename))
print("If you don't want to continue,please press ctrl+c")
print("if you want,press return")

input("?")

print("Opening the file...")
target = open(filename,'w')

# print("Truncating the file. Goodbye")
# target.truncate()

print("Now I'm going to ask you for three lines")

line1 = input("line1:")
line2 = input("line2:")
line3 = input("line3:")


target.write(line1)
# target.write(line1+"\n"+line2+"\n"+line3+"\n")

print("and finally,we close it")
target.close()
