import numpy as np

class Cube:
    def __init__(self):
        self.upper = np.array([[0,0,0],[0,0,0],[0,0,0]]) #white
        self.front = np.array([[1,1,1],[1,1,1],[1,1,1]]) #orange
        self.lower = np.array([[2,2,2],[2,2,2],[2,2,2]]) #yellow
        self.right = np.array([[3,3,3],[3,3,3],[3,3,3]]) #green
        self.left  = np.array([[4,4,4],[4,4,4],[4,4,4]]) #blue
        self.back  = np.array([[5,5,5],[5,5,5],[5,5,5]]) #red

    def display(self):
        print (self.upper)
        print("upper")
        print('\n')
        print (self.front)
        print("upper")
        print('\n')
        print (self.lower)
        print("lower")
        print('\n')
        print (self.right)
        print("right")
        print('\n')
        print (self.left )
        print("left")
        print('\n')
        print (self.back)
        print("back")
        print('\n')

    def R(self):
        temp = self.upper[:,2].copy()

        self.upper [:,2] = self.front[:,2]
        self.front [:,2] = self.lower[:,2]
        self.lower [:,2] = np.flip(self.back[:,2])
        self.back [:,2] = np.flip(temp)

        self.right = np.rot90(self.right,-1);

    def R_Prime(self):
        self.R()
        self.R()
        self.R()

    def L(self):
        temp = self.upper[:,0].copy()

        self.upper [:,0] = self.front[:,0]
        self.front [:,0] = self.lower[:,0]
        self.lower [:,0] = np.flip(self.back[:,0])
        self.back [:,0] = np.flip(temp)

        self.left = np.rot90(self.right,1);

    def L_prime(self):
        self.L()
        self.L()
        self.L()