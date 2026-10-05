import numpy as np

# note: every face is stored as seen from OUTSIDE the cube.
# upper: row0 = back side, row2 = front side
# front/right/left/back/lower: standard "net" orientation
class Cube:
    def __init__(face):
        face.upper = np.array([[0,0,0],[0,0,0],[0,0,0]]) #white
        face.front = np.array([[1,1,1],[1,1,1],[1,1,1]]) #orange
        face.lower = np.array([[2,2,2],[2,2,2],[2,2,2]]) #yellow
        face.right = np.array([[3,3,3],[3,3,3],[3,3,3]]) #green
        face.left  = np.array([[4,4,4],[4,4,4],[4,4,4]]) #blue
        face.back  = np.array([[5,5,5],[5,5,5],[5,5,5]]) #red

    def display(face):
        print (face.upper)
        print("upper")
        print('\n')
        print (face.front)
        print("front")
        print('\n')
        print (face.lower)
        print("lower")
        print('\n')
        print (face.right)
        print("right")
        print('\n')
        print (face.left )
        print("left")
        print('\n')
        print (face.back)
        print("back")
        print('\n')

    def R(face):
        temp = face.upper[:,2].copy()

        face.upper [:,2] = face.front[:,2]
        face.front [:,2] = face.lower[:,2]
        face.lower [:,2] = np.flip(face.back[:,0])
        face.back  [:,0] = np.flip(temp)

        face.right = np.rot90(face.right,-1)

    def R_Prime(face):
        face.R()
        face.R()
        face.R()

    def L(face):
        temp = face.upper[:,0].copy()

        face.upper [:,0] = np.flip(face.back[:,2])
        face.back  [:,2] = np.flip(face.lower[:,0])
        face.lower [:,0] = face.front[:,0]
        face.front [:,0] = temp

        face.left = np.rot90(face.left,-1)

    def L_Prime(face):
        face.L()
        face.L()
        face.L()

    def U(face):
        temp = face.front[0,:].copy()

        face.front[0,:] = face.right[0,:]
        face.right[0,:] = face.back[0,:]
        face.back [0,:] = face.left[0,:]
        face.left [0,:] = temp

        face.upper = np.rot90(face.upper,-1)

    def U_Prime(face):
        face.U()
        face.U()
        face.U()

    def D(face):
        temp = face.front[2,:].copy()

        face.front[2,:] = face.left[2,:]
        face.left [2,:] = face.back[2,:]
        face.back [2,:] = face.right[2,:]
        face.right[2,:] = temp

        face.lower = np.rot90(face.lower,-1)

    def D_Prime(face):
        face.D()
        face.D()
        face.D()

    def F(face):
        temp = face.upper[2,:].copy()

        face.upper[2,:] = np.flip(face.left[:,2])
        face.left [:,2] = face.lower[0,:]
        face.lower[0,:] = np.flip(face.right[:,0])
        face.right[:,0] = temp

        face.front = np.rot90(face.front,-1)

    def F_Prime(face):
        face.F()
        face.F()
        face.F()

    def B(face):
        temp = face.upper[0,:].copy()

        face.upper[0,:] = face.right[:,2]
        face.right[:,2] = np.flip(face.lower[2,:])
        face.lower[2,:] = face.left[:,0]
        face.left [:,0] = np.flip(temp)

        face.back = np.rot90(face.back,-1)

    def B_Prime(face):
        face.B()
        face.B()
        face.B()