#coding: utf-8
import sys, os
from storage import Storage

if __name__ == "__main__":
    from dataset.cityscapes import CityScapes
    c = CityScapes()
    print(c.get_coarse_img())