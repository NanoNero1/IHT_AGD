"""Desc: a simple script to see if we can write to the storage on the cluster"""
print(__file__)

import os
cwd = os.getcwd()
print(cwd)

with open('readme.txt', 'w') as f:
    f.write('Create a new text file!')