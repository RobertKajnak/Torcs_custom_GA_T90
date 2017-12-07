#! /usr/bin/env python3
import sys
sys.path.insert(0, '/home/student/Documents/torcs-server/torcs-client')
from my_driver_evo import MyDriver
from pytocl.main import main

#import argparse


if __name__ == '__main__':
    main(MyDriver())


