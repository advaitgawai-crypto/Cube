from notations import Cube
import numpy as np
import random as rd


def algorithm_1(cube: Cube):
    cube.R_Prime()
    cube.D()
    cube.R()

def algorithm_2(cube: Cube):
    cube.R_Prime()
    cube.D_Prime()
    cube.R()

def algorithm_3(cube: Cube):
    cube.L_Prime()
    cube.D_Prime()
    cube.L()

def algorithm_4(cube: Cube):
    cube.L_Prime()
    cube.D()
    cube.L()

def algorithm_5(cube: Cube):
    cube.Y()

def algorithm_6(cube: Cube):
    cube.Y_prime()



