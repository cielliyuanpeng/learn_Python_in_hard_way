from sys import argv

script,username = argv
prompt = ">"

print("hi,{},I'm the {} script".format(username,script))
print("I'd like to ask you some questions")
like = input("Do you like me?\n{}".format(prompt))
lives = input("Where do you live\n"+prompt)
computer = input(f"what kind of computer do you have?\n{prompt}")
print(f"""
Alright, so you said {like} about me.
you lives in {lives}.
you have a {computer},nice
""")
