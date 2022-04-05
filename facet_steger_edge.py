import numpy as np
import cv2
from chain_edge import deal
import math


img = cv2.imread("E:/power_line/dalunwen/calibresult7.jpg",0)
# edge = deal(img)
edge = cv2.imread("E:/power_line/dalunwen/testpic.jpg",0)

img = cv2.GaussianBlur(img,(5,5),2)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
tpl_gx = cv2.Sobel(img,cv2.CV_32F,1,0,3)
tpl_gy = cv2.Sobel(img,cv2.CV_32F,0,1,3)
# edge = cv2.Canny(img,50,150)

contours,hireachy = cv2.findContours(edge,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
mag_tpl = cv2.magnitude(tpl_gx,tpl_gy)

def magNeighbour(x,y,mag):
    h,w=mag_tpl.shape
    top = y - 1 if y - 1 >=0 else y
    down = y + 1 if y + 1 < h else y
    left = x - 1 if x - 1 >= 0 else x
    right = x + 1 if x + 1 < w else x
    mag[0]=mag_tpl[top][left]
    mag[1]=mag_tpl[top][x]
    mag[2]=mag_tpl[top][right]
    mag[3]=mag_tpl[y][left]
    mag[4]=mag_tpl[y][x]
    mag[5]=mag_tpl[y][right]
    mag[6]=mag_tpl[down][left]
    mag[7]=mag_tpl[down][x]
    mag[8]=mag_tpl[down][right]
    
def get2ndFacetModel(mag,a):
    # a[0] = (-mag[0] + 2.0 * mag[1] - mag[2] + 2.0 * mag[3] + 5.0 * mag[4] + 2.0 * mag[5] - mag[6] + 2.0 * mag[7] - mag[8]) / 9.0
    # a[1] = (-mag[0] + mag[2] - mag[3] + mag[5] - mag[6] + mag[8]) / 6.0
    # a[2] = (mag[6] + mag[7] + mag[8] - mag[0] - mag[1] - mag[2]) / 6.0
    # a[3] = (mag[0] - 2.0 * mag[1] + mag[2] + mag[3] - 2.0 * mag[4] + mag[5] + mag[6] - 2.0 * mag[7] + mag[8]) / 6.0
    # a[4] = (-mag[0] + mag[2] + mag[6] - mag[8]) / 4.0
    # a[5] = (mag[0] + mag[1] + mag[2] - 2.0 * (mag[3] + mag[4] + mag[5]) + mag[6] + mag[7] + mag[8]) / 6.0
    a[0] = (mag[0] +  mag[1] + mag[2] +  mag[3] +  mag[4] +  mag[5] + mag[6] +  mag[7] + mag[8]) / 9.0
    a[1] = (-mag[0] - mag[1] - mag[2] + mag[7] + mag[6] + mag[8]) / 6.0
    a[2] = (mag[2] -  mag[0] - mag[3] + mag[5] -  mag[6]  + mag[8]) / 6.0
    a[3] = (-mag[0] -  mag[1] - mag[2] -2.0*  mag[3] -2.0*  mag[4] -2.0*  mag[5] - mag[6] -  mag[7] - mag[8]) / 18.0
    a[4] = ( mag[0]  - mag[2]  - mag[6] + mag[8]) / 4.0
    a[5] = (mag[0] - mag[2] + 2.0 * (mag[3] - mag[5]) + mag[6] - mag[8]) / 12.0
    a[6] = -2 * (-mag[0] - mag[1] - mag[2] + mag[7] + mag[6] + mag[8]) / 24.0
    a[7] = (mag[0] + 2 * mag[1] + mag[2] -2* mag[7] - mag[6] - mag[8]) / 8.0
    a[8] = -2 * (mag[2] -  mag[0] - mag[3] + mag[5] -  mag[6]  + mag[8]) / 24.0

def eigenvals(a,eigval,eigvec):
    dfdrc = a[4]
    dfdcc = a[3] * 2.0
    dfdrr = a[5] * 2.0
    theta, t, c, s, e1, e2, n1, n2 = 0,0,0,0,0,0,0,0
    if dfdrc != 0.0:
        theta = 0.5*(dfdcc - dfdrr) / dfdrc
        t = 1.0 / (abs(theta) + np.sqrt(theta*theta + 1.0))
        if theta < 0.0:
            t = -t
        c = 1.0 / np.sqrt(t*t + 1.0)
        s = t*c
        e1 = dfdrr - t*dfdrc
        e2 = dfdcc + t*dfdrc
    else:
        c = 1.0
        s = 0.0
        e1 = dfdrr
        e2 = dfdcc
    n1 = c
    n2 = -s
    hessian_mat = np.array([[dfdcc,dfdrc],[dfdrc,dfdrr]])
    a_val, b_vec = np.linalg.eig(hessian_mat)
    
    if abs(e1) > abs(e2):
        eigval[0] = e1
        eigval[1] = e2
        eigvec[0][0] = n1
        eigvec[0][1] = n2
        eigvec[1][0] = -n2
        eigvec[1][1] = n1
    elif abs(e1) < abs(e2):
        eigval[0] = e2
        eigval[1] = e1
        eigvec[0][0] = -n2
        eigvec[0][1] = n1
        eigvec[1][0] = n1
        eigvec[1][1] = n2
    else:
        if e1 < e2:
            eigval[0] = e1
            eigval[1] = e2
            eigvec[0][0] = n1
            eigvec[0][1] = n2
            eigvec[1][0] = -n2
            eigvec[1][1] = n1
        else:
            eigval[0] = e2
            eigval[1] = e1
            eigvec[0][0] = -n2
            eigvec[0][1] = n1
            eigvec[1][0] = n1
            eigvec[1][1] = n2
    # if abs(a_val[0])>abs(a_val[0]):
    #     eigvec[0] = b_vec[0]
    #     eigvec[1] = b_vec[1]
    # else:
    #     eigvec[0] = b_vec[1]
    #     eigvec[1] = b_vec[0]
    # eigvec = b_vec
    # print("eigval:",eigvec)
    # print(np.sqrt(eigvec[0][0]**2+eigvec[0][1]**2))

icontour = []
left=[]
right=[]
h1=[]
h2=[]
v1=[]
v2=[]
for i in range(len(contours)):
    point = []
    for j in range(len(contours[i])):
        mag = np.zeros(9)
        magNeighbour(contours[i][j][0][0],contours[i][j][0][1],mag)
        a = np.zeros(9)
        get2ndFacetModel(mag,a)
        
        eigvec = np.zeros((2,2))
        eigval = np.zeros(2)
        eigenvals(a,eigval,eigvec)
        t = 0.0
        ny = eigvec[0][0]
        nx = eigvec[0][1]
        if eigval[0] < 0:
            rx = a[1] #-1*a[6]- 1/3*a[7]
            ry = a[2] #- 1/3 *a[5] -1*a[8]
            rxy = a[4]
            rxx = a[3] * 2.0
            ryy = a[5] * 2.0
            t = -(rx * nx + ry * ny) / (rxx * nx * nx + 2.0 * rxy * nx * ny + ryy * ny * ny)
        
        px = nx * t
        py = ny * t
        xx = contours[i][j][0][0]
        yy = contours[i][j][0][1]
        
        if abs(px) <= 0.5 and abs(py) <= 0.5:
            xx = xx + px
            yy = yy + py
            
        
        
        point.append([xx,yy])
        

    icontour.append(point)



# cv2.imshow("img",img1)
# cv2.waitKey(0)
cv2.imwrite("tt1.jpg",edge)

