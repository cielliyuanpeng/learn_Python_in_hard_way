mystuff = ['socks','pen','books']
mystuff.append('mouse')
print(mystuff)

class Thing(object):
    def test(self,message):
        print(message)

a = Thing()
a.test('aaaaaaaa')

ten_things = "Apple Oranges Crows Telephone Light Sugar"

print('Wait ,add things')
stuff = ten_things.split(' ')
more_stuff = ['1','2','3','4','5']
while len(stuff)<10:
    next_one = more_stuff.pop()
    print("Adding:",next_one)
    stuff.append(next_one)
    print(f"There are {len(stuff)} items now")

print('There we go',stuff)
print(stuff[-1])
print(stuff[1])
print(stuff.pop())
print(' '.join(stuff))
print('#'.join(stuff[:]))#左闭右开
