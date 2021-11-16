print("Lets practice everything")
print('You\'d need to know\'bout escapes with \\ that do:')
print(" \n newline and \t tabs")

poem = """
\t The lovely world
with logic so firmly planted
cannot discen \n the needs of live
nor comprehend passion from intuition
and requires an explanation
\n\t\twhere there  is none
"""

print("-----------")
print(poem)
print("-----------")

five = 10-2+3-6

print(f"This should be five:{five}")

def secret_fomula(started):
    jelly_beans = started * 500
    jars = jelly_beans / 1000
    crates = jars / 100
    return jelly_beans,jars,crates #python 的函数能返回多个值？

start_point = 10000

beans,jars,crates = secret_fomula(start_point)

print("with a start point of:{}".format(start_point))
print(f"we'd have {beans} beans,{jars} jars,and {crates} crates")

start_point = start_point/10

print("we can also do this way")
formula = secret_fomula(start_point)
print("we have {} beans,{} jars, and {} crats.".format(*formula))
