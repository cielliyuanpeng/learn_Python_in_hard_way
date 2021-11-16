def append_i_to_num(max):
    i = 0
    numbers = []
    while i<max:
        print(f"At the top i is {i}")
        numbers.append(i)
        i += 1
        print("Numbers now:",numbers)
        print(f"At the bottom i is {i}")

    return numbers

numbers = append_i_to_num(int(input("how many do you wannna append?")))
print("the numbers: ")
for num in numbers:
    print(num,end = " ")
