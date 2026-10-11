from notations import Cube
import numpy as np
import random as rd

class CrossSafe:
    def algorithm_1(self, cube: Cube):
        cube.R_Prime()
        cube.D()
        cube.R()    

    def algorithm_2(self, cube: Cube):
        cube.R_Prime()
        cube.D_Prime()
        cube.R()

    def algorithm_3(self, cube: Cube):
        cube.L_Prime()
        cube.D_Prime()
        cube.L()

    def algorithm_4(self, cube: Cube):
        cube.L_Prime()
        cube.D()
        cube.L()

    def algorithm_5(self, cube: Cube):
        cube.Y()

    def algorithm_6(self, cube: Cube):
        cube.Y_prime()



