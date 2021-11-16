from sys import exit

def gold_room():
    print("这是一间装满黄金的屋子，你会拿多少呢？")

    choice = input(">")
    if choice.isnumeric():
        how_much = int(choice)
    else:
        dead("输入的不是数字，san值过低")

    if how_much < 50:
        print("太好了，你战胜了心中的贪念，获得了重生")
        exit(0)
    else:
        dead("你被心中的贪念诱惑，陷入万劫不复之地")

def bear_room():
    print("一只熊坐在餐桌前用勺子吃午餐")
    bear_moved = False
    print("你要怎么做呢？")
    while True:
        choice = input(">")
        if choice == "掀桌子":
            print("熊被你吓走了，你赢了,走进了下一个房间")
            gold_room()
        if choice == "打熊一拳" and not bear_moved:
            bear_moved = not bear_moved
            print("熊被你吓了一跳")
            print("接下来怎么做呢？")
        elif choice == "打熊一拳" and bear_moved:
            dead("熊觉得受到了侮辱，KO了你")
        else:
            print("你的动作熊视而不见，没起到效果")
            print("接下来怎么做呢？")

def dead(why):
    print(why,"再见")
    exit(0)


bear_room()
