from notations import Cube 
import numpy as np
# order: top --> front --> bottom --> back --> left --> right

cube = Cube()

def flatten_cube(cube):
    flatFront = cube.front.flatten()
    flatUpper = cube.upper.flatten()
    flatLower = cube.lower.flatten()
    flatLeft = cube.left.flatten()
    flatRight = cube.right.flatten()
    flatBack = cube.back.flatten()
    flattenedCube = np.concatenate([flatUpper,flatFront,flatLower,flatBack,flatLeft,flatRight])
    return flattenedCube

def unflatten_cube(a):
    cube = Cube()
    cube.upper = a[0:9].reshape(3 , 3).copy()
    cube.front = a[9:18].reshape(3 ,3).copy()
    cube.lower = a[18:27].reshape(3,3).copy()
    cube.back  = a[27:36].reshape(3,3).copy()
    cube.left  = a[36:45].reshape(3,3).copy()
    cube.right = a[45:54].reshape(3,3).copy()    
    return cube