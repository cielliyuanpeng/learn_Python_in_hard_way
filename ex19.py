def cheese_and_crackers(cheese_count,boxex_of_crackers):
    print("you have {} cheesses".format(cheese_count))
    print("you have {} boxes of crackers".format(boxex_of_crackers))
    print("get a blanket\n")

print("We can give the function N directly")
cheese_and_crackers(20,30)

print("or print variable in script")

amount_of_cheese = 10;
amount_of_crackers = 50
cheese_and_crackers(amount_of_cheese,amount_of_crackers)

print("we even can do math")
cheese_and_crackers(10+20,5+6)

print("we can combine the two,variable and math")
cheese_and_crackers(amount_of_cheese+100,amount_of_crackers+1000)
