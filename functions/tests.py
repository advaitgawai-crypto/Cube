from notations import Cube

cube = Cube()
cube.display()
print('\n\n\n')
for _ in range(1):
    cube.R()
    cube.L()
    cube.U()
    cube.D()
    cube.F()
    cube.B()
cube.display()
