import numpy as np
import cv2
from scipy import optimize
import matplotlib.pyplot as plt
import math

Xyzs = np.loadtxt('1.txt')
Rgbs = np.loadtxt('2.txt')
Xyzs_R = Xyzs[0:17].T
Xyzs_G = Xyzs[17:34].T
Xyzs_B = Xyzs[34:51].T
rgb_sequence = Rgbs[0:17].T
rgb_sequence = rgb_sequence[0]
# print(rgb_sequence)
# print(Xyzs_R)

#Xw是R通道255的X值，Yw是G通道255的Y值，Zw是B通道255的Z值
Xw = 78.562
Yw = 71.689
Zw = 107.72

#黑点
Xk = 0.12141
Yk = 0.12823
Zk = 0.13531


def func1(x, p):
    kg, k0, gama = p
    return np.power(kg * x / 255 + k0, gama)
def GoG(K, k1, ds):
    L = K/k1
    def func(x,p):
        kg, k0, gama = p
        return np.power(kg*x/255+k0, gama)

    def res(p,y,x):
        #print(p)
        return y - func(x,p)

    plsq = optimize.leastsq(res, np.array([1, 0.2, 0]), args=(L, ds))
    kg, k0, gama = plsq[0]
    print(plsq)
    d0 = func(ds, plsq[0]) - L
    d1 = np.sum(d0*d0)/d0.shape
    return kg, k0, gama, d1

def reverse_GoG(x, p):
    kg, k0, gama = p
    return (math.pow(x, 1/gama)-k0)*(255/kg)

# atps = 100
# save_box = np.zeros([atps,4])
# for i in range(atps):
#     save_box[i] = GoG(Xyzs_G[0], Yw, rgb_sequence)
# save_box2 = save_box.T
# save_box3 = save_box2[]

pr1 = GoG(Xyzs_R[0], Xw, rgb_sequence)
pR = np.array([pr1[0], pr1[1],pr1[2]])
pg1 = GoG(Xyzs_G[1], Yw, rgb_sequence)
pG = np.array([pg1[0], pg1[1],pg1[2]])
pb1 = GoG(Xyzs_B[2], Zw, rgb_sequence)
pB = np.array([pb1[0], pb1[1],pb1[2]])

xx = np.linspace(0, 255, 200)
yy = func1(xx,pB)
plt.plot(xx, yy,'r')
plt.plot(rgb_sequence, Xyzs_B[2]/Zw)
#plt.show()

def forward_translation(rgb, pR, pG, pB):
    d_matrix = np.array([[78.562,39.674,1.5915],[22.336,71.689,11.82],[20.251,10.03,107.72]]).T
    lr = func1(rgb[0],pR)
    lg = func1(rgb[1],pG)
    lb = func1(rgb[2],pB)
    l_matrix = np.array([lr,lg,lb])
    xyz0 = np.array([Xk,Yk,Zk])
    xyzs = np.dot(d_matrix, l_matrix)+xyz0
    return xyzs

def backward_translation(xyz, pR,pG,pB):
    d_matrix = np.array([[78.562,39.674,1.5915],[22.336,71.689,11.82],[20.251,10.03,107.72]]).T
    d_reverse = np.linalg.inv(d_matrix)
    xyz0 = np.array([Xk, Yk, Zk]).T
    xyz1 = np.array([xyz[0], xyz[1], xyz[2]]).T
    xyz2 = xyz1 - xyz0
    L = np.dot(d_reverse, xyz2)
    if L[0]<0:
        L[0]=0
    if L[1]<0:
        L[1]=0
    if L[2]<0:
        L[2]=0
    r = reverse_GoG(L[0], pR)
    g = reverse_GoG(L[1], pG)
    b = reverse_GoG(L[2], pB)
    rgb = np.array([r,g,b])
    return rgb

def xyz2lab(xyzs):
    labs=np.zeros((xyzs.shape[0],3))
    Xn = 122.27
    Yn = 122.47
    Zn = 121.61
    for i in range(xyzs.shape[0]):
        if xyzs[i][0]/Xn > math.pow(24/116,3) :
            fX = math.pow(xyzs[i][0]/Xn, 1/3)
        else:
            fX = (841/108)*(xyzs[i][0]/Xn)+16/116
        if xyzs[i][1]/Yn > math.pow(24/116,3):
            fY = math.pow(xyzs[i][1]/Xn, 1/3)
        else:
            fY = (841/108)*(xyzs[i][1]/Yn)+16/116
        if xyzs[i][2]/Zn > math.pow(24/116,3):
            fZ = math.pow(xyzs[i][2]/Xn, 1/3)
        else:
            fZ = (841/108)*(xyzs[i][2]/Zn)+16/116
        labs[i][0] = 116*fY-16
        labs[i][1] = 500*(fX-fY)
        labs[i][2] = 200*(fY-fZ)
    return labs

def Eabs(lab1, lab2):
    dL = lab1[0]-lab2[0]
    da = lab1[1]-lab2[1]
    db = lab1[2]-lab2[2]
    Eabs = math.pow(math.pow(dL, 2) + math.pow(da, 2) + math.pow(db, 2),0.5)
    return Eabs

test_daset = np.loadtxt('3.txt')
rgb_box = np.zeros([test_daset.shape[0],3])
for i in range(test_daset.shape[0]):
    rgb_box[i] = backward_translation(test_daset[i], pR,pG,pB)

print(rgb_box)

list = []
for i in range(rgb_box.shape[0]):
    if 0<rgb_box[i][0]<255 and 0<rgb_box[i][1]<255 and 0<rgb_box[i][2]<255:
        list.append(i)

print(list)
#
# with open('4-1.txt', 'w') as f:
#     for i in list:
#         a = rgb_box[i][0]
#         b = rgb_box[i][1]
#         c = rgb_box[i][2]
#         a1 = format(a, '.0f')
#         b1 = format(b, '.0f')
#         c1 = format(c, '.0f')
#         f.write(a1)
#         f.write('\t')
#         f.write(b1)
#         f.write('\t')
#         f.write(c1)
#         f.write('\n')
#
# with open('4-2.txt', 'w') as f:
#     for i in list:
#         a = test_daset[i][0]
#         b = test_daset[i][1]
#         c = test_daset[i][2]
#         a1 = format(a, '.0f')
#         b1 = format(b, '.0f')
#         c1 = format(c, '.0f')
#         f.write(a1)
#         f.write('\t')
#         f.write(b1)
#         f.write('\t')
#         f.write(c1)
#         f.write('\n')

rgb_origin = np.loadtxt('4-1.txt')
pre_xyz = np.zeros([rgb_origin.shape[0],3])
for i in range(pre_xyz.shape[0]):
    pre_xyz[i] = forward_translation(rgb_origin[i], pR, pG, pB)

predict_xyz = np.loadtxt('4-2.txt')
groundtruth_xyz = np.loadtxt('4-3.txt')
gt_lab = xyz2lab(groundtruth_xyz)
pre_lab = xyz2lab(predict_xyz)
eab_box = np.zeros((pre_lab.shape[0]))
for i in range(pre_lab.shape[0]):
    eab_box[i] = Eabs(gt_lab[i], pre_lab[i])

print(eab_box)
print(eab_box.max())
print(eab_box.min())
print(eab_box.mean())






