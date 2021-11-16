# -*- coding: utf-8 -*-

from sys import argv
script,filename = argv
txt = open(filename)
print(txt.read())
file_again = input(">")
txt = open(file_again)
print(txt.read())
txt.close()
