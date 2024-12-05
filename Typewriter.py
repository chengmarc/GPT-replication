# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import sys, time


# %%
def typewrite(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)


# %%
if __name__ == "__main__":
    typewrite("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean et tristique tellus. \
              Aliquam malesuada at odio vel porttitor. Donec id lectus mattis, venenatis elit at, fermentum lectus.")

