def print_two(*args):# 把所有参数放进一个列表中
    arg1,arg2 = args
    print("arg1: {} arg2: {}".format(arg1,arg2))

def print_two_again(arg1,arg2):
    print("arg1: {},arg2: {}".format(arg1,arg2))

def print_none():
    print("I got nothing")

print_two("1",2)
print_two_again("lij","IJ")
print_none()
