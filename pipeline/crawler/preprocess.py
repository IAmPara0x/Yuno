#! /usr/bin/python3

with open("proxy-list.txt", "r+") as f:
  x = f.readlines()
  f.seek(0)
  x = [i.split("]")[1].split(">")[0][1:] for i in x]
  x = ["http://"+i+"\n" for i in x]
  f.writelines(x)
  f.truncate()
