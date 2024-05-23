import mujoco_py as mj
import numpy as np
from envs.rotations import mat2quat, mat2embedding, quat2mat
from pathlib import Path
import time
from matplotlib import pyplot as plt


class value(): #value class defined to set and copy mujoco states
    def __init__(self,time,qpos,qvel,act,udd_state):
        self.time=time
        self.qpos=qpos
        self.qvel=qvel
        self.act=act
        self.udd_state=udd_state


def setp(a: np.ndarray, b: np.ndarray): #sets mocap position and quaternion to the input vectors a and b
    sim.data.mocap_pos[0][:]=a
    sim.data.mocap_quat[0][:]=b

def noth(sim: mj.MjSim, view: mj.MjViewerBasic, num: int): #runs simulation for num steps
    for i in range(num):
        #time.sleep(0.04)
        sim.step()
        sim.data.qvel[:]=0.0
        mj_viewer.render()

def calcAng(x: np.ndarray):
    #calculates the parameters phi and theta for given contact point relative to its origin (calculated in getorigin1) in our variant of sphere coordinates
    phi=np.arctan2(x[1],-x[0])
    phi=phi%(2*np.pi)
    phi=phi.item()
    theta=np.arccos(x[2])
    theta=theta.item()
    #print("phi: ", phi.item())
    #print("theta: ",theta)
    return phi,theta

def pos(phi: float,theta: float):
    #calculates a point on the unit sphere for given parameters in our variant of sphere coordinates
    if type(phi)==np.ndarray:
        phi=phi.item()
    if type(theta)==np.ndarray:
        theta=theta.item()
    #print("phi= ", phi, "theta= ", theta)
    r=np.array([-np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).reshape(1,-1)
    r=r.T
    return r

def grad(phi: float, theta: float):
    #returns gradients of the contact point in our variant of sphere coordinates
    #gphi=np.array([[np.sin(phi)], [np.cos(phi)], [0.0]])#.reshape(1,-1)
    #gphi=gphi.T
    gphi=np.array([np.sin(phi), np.cos(phi), 0.0], dtype=object).reshape(1,-1)
    gphi=gphi.T
    gphi[np.absolute(gphi)<1e-5]=0
    gtheta=np.array([-np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)], dtype=object).reshape(1,-1)
    gtheta=gtheta.T
    gtheta[np.absolute(gtheta)<1e-5]=0
    return gphi, gtheta

def getorigin1(p: np.ndarray):
    #returns sphere origin for given contact position such that this contact point lies on the sphere surface
    mx1=np.max(p)
    i1=np.argmax(p)
    mi1=np.min(p)
    j1=np.argmin(p)
    o=np.copy(p)
    #print("mx= ", mx1)
    if mx1>=0.04 and mi1>0.0001:
        o[i1]=o[i1]-1

    elif mi1 <=0.008:
        o[j1]=o[j1]+1
    
    return o

def proj1(p: np.ndarray):
    #projects point p, which is on a sphere onto the closest object (cube) surface
    mx=np.max(np.absolute(p))
    i=np.argmax(np.absolute(p))
    mn=np.min(np.absolute(p))
    j=np.argmin(np.absolute(p))
    
    b=p
    if 0.05-mx<mn and np.sign(p[j])==1:
        b[i]=np.sign(p[i])*0.05
    else:
        b[j]=0
    
    return b

def proj2(p: np.ndarray):
    #another form of surface projection, used in case there are negative values
    mx=np.max(p)
    i=np.argmax(p)
    mn=np.min(p)
    j=np.argmin(p)
    b=p
    if 0.05-mx<mn and np.sign(p[j])==1:
        b[i]=np.sign(p[i])*0.05
    else:
        b[j]=0
    
    return b

def proj3(p: np.ndarray):#checks if contact point position is outside the cube boundaries (xyz values must lie between 0 and 0.05) and projects them into the cube boundaries
    b=p
    c=0
    for i in range(3):
        if p[i]>0.05 or p[i]<0:
            c+=1
    
    if c>1:
        for i in range(3):
            if p[i]>0.05:
                b[i]=0.05

            elif p[i]<0.05:
                b[i]=0

    return b

def getM(p1: np.ndarray, p2: np.ndarray): #calculates torque tau for moment residual control
    c=0.5*(p1+p1)
    #print("centroid: ", c)
    r1=p1-0.5*(p1+p2)#p1-c
    #print("r1 = ", r1)
    r2=p2-0.5*(p1+p2)#p2-c
    #print("r2 = ", r2)
    n1=getn(p1)
    #print(n1)
    n2=getn(p2)
    #print(n2)
    m=np.cross(r1.T,n1.T) + np.cross(r2.T,n2.T)
    m=m.T
    return m

def Roty(theta: float):
    #returns Rotationmatrix for a rotation of theta around the y-axis
    #this function is used for a special case in the moment residual control algorithm
    R=np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return R

def getF(p1: np.ndarray, p2: np.ndarray):
    #returns total force on the object, used for force residual control
    n1=getn(p1)
    n2=getn(p2)
    F=n1+n2
    return F

def getn(p: np.ndarray):
    #returns corresponding object surface normal for given contact point 
    mx1=np.max(p)
    i1=np.argmax(p)
    mi1=np.min(p)
    j1=np.argmin(p)
    n=np.array([0,0,0])
    if mx1>=0.04 and mi1>0.0001:
        n[i1]=1
        n=n.reshape(1,-1)
        n=n.T

    elif mi1<=0.011:
        n[j1]=-1
        n=n.reshape(1,-1)
        n=n.T

    return n

def checksingularity(p1: np.ndarray, n1: np.ndarray, p2: np.ndarray, n2: np.ndarray):
    #checks whether contact points are in an alternating configuration as in figure 4.1 . Returns boolean value s. If s=1 force residual control runs one iteration on only one finger while the other remains stationary
    i1=np.argmax(np.absolute(n1))
    i2=np.argmax(np.absolute(n2))
    a=p1[i2].item()
    ac=0.05-a
    b=p2[i1].item()
    #print(np.absolute(ac-b))
    bc=0.05-b
    if i1!=i2:
        if np.absolute(a-b)<3e-3 or np.absolute(ac-b)<3e-3 or np.absolute(a-bc)<3e-3:
            s=1
        else:
            s=0
    
    else:
        s=0
    
    return s

def Moment(p1: np.ndarray, p2: np.ndarray):

    #Conducts one iteration of moment residual control

    # calculates sphere origins, surface normals, total torque m
    o1=getorigin1(p1)
    o2=getorigin1(p2)
    
    n1=getn(p1)
    n2=getn(p2)
    m=getM(p1,p2)
    gamma=-0.08
    #checks for special case where contact normals are in z direction
    if np.linalg.norm(np.absolute(n1)-np.array([[0],[0],[1]]))<1e-4 or np.linalg.norm(np.absolute(n2)-np.array([[0],[0],[1]]))<1e-4:
        if ((np.linalg.norm(np.absolute(n1)-np.array([[0],[0],[1]]))<1e-4 and np.linalg.norm(np.absolute(n2)-np.array([[0],[0],[1]]))<1e-4) or (np.linalg.norm(np.absolute(n1)-np.array([[0],[0],[1]]))<1e-4 and np.linalg.norm(np.absolute(n2)-np.array([[0],[1],[0]]))<1e-4) or  (np.linalg.norm(np.absolute(n2)-np.array([[0],[0],[1]]))<1e-4 and np.linalg.norm(np.absolute(n1)-np.array([[0],[1],[0]]))<1e-4)):
            #checks whether both contacts go in z direction or if one goes in z and the other in y direction. If that is the case the rotation matrix is defined as roty(270)

            R=roty(270)
        elif (np.linalg.norm(np.absolute(n1)-np.array([[0],[0],[1]]))<1e-4 and np.linalg.norm(np.absolute(n2)-np.array([[1],[0],[0]]))<1e-4 ) or (np.linalg.norm(np.absolute(n2)-np.array([[0],[0],[1]]))<1e-4 and np.linalg.norm(np.absolute(n1)-np.array([[1],[0],[0]]))<1e-4):
            #checks whether one contact has normal in z direction and other one in x direction, In this case the rotation matrix is defined as rotx(270)
            R=rotx(270)
        #print(R)
        nn1=R@n1
        nn2=R@n2
        mm=R@m
        phi1,theta1=calcAng(nn1)
        phi2,theta2=calcAng(nn2)
        u=np.array([[phi1],[theta1],[phi2],[theta2]]) #define combined parameter vector u
        gradphi1,gradtheta1=grad(phi1,theta1)
        gradphi2,gradtheta2=grad(phi2,theta2)
        #print(nn1.T)
        #print(nn2.T)
        J=np.zeros((4,3))
        # in gerneral u has the form [u1, v1, u2, v2]
        #The following cases determine ui and vi
        if np.linalg.norm(gradphi1-gradphi2)<1e-4:
            
            a=0
            u=np.array([[theta1],[phi1],[theta2],[phi2]])
            gradV=gradphi1
            gradU1=gradtheta1
            gradU2=gradtheta2
            J=np.concatenate((-gradV.T, gradU1.T, -gradV.T, gradU2.T))
        elif np.linalg.norm(gradtheta1-gradtheta2)<1e-4:
            #print("p2= ", p2)
            a=1
            u=np.array([[phi1],[theta1],[phi2],[theta2]])
            gradV=gradtheta1
            gradU1=gradphi1
            gradU2=gradphi2
            J=np.concatenate((-gradV.T, gradU1.T, -gradV.T, gradU2.T))
        
        #print("Jacobian J= ",J)
        udot=J@mm
        #print("udot= ", udot)
        u1=u+gamma*udot
        p1new=pos(u1[0],u1[1])
        p2new=pos(u1[2],u1[3])
        p1new=R.T@p1new
        p2new=R.T@p2new

    # elif (np.linalg.norm(np.absolute(n1)-np.array([[0],[0],[1]]))<1e-4 and np.linalg.norm(np.absolute(n2)-np.array([[1],[0],[0]]))<1e-4 ) or (np.linalg.norm(np.absolute(n2)-np.array([[0],[0],[1]]))<1e-4 and np.linalg.norm(np.absolute(n1)-np.array([[1],[0],[0]]))<1e-4):
    #         R=rotx(270)
    #         #print(R)
    #         nn1=R@n1
    #         nn2=R@n2
    #         mm=R@m
    #         phi1,theta1=calcAng(nn1)
    #         phi2,theta2=calcAng(nn2)
    #         u=np.array([[phi1],[theta1],[phi2],[theta2]])
    #         gradphi1,gradtheta1=grad(phi1,theta1)
    #         gradphi2,gradtheta2=grad(phi2,theta2)
    #         J=np.zeros((4,3))

    #         if np.linalg.norm(gradphi1-gradphi2)<1e-4:
                
    #             a=0
    #             u=np.array([[theta1],[phi1],[theta2],[phi2]])
    #             gradV=gradphi1
    #             gradU1=gradtheta1
    #             gradU2=gradtheta2
    #             J=np.concatenate((-gradV.T, gradU1.T, -gradV.T, gradU2.T))
    #         elif np.linalg.norm(gradtheta1-gradtheta2)<1e-4:
    #             #print("p2= ", p2)
    #             a=1
    #             u=np.array([[phi1],[theta1],[phi2],[theta2]])
    #             gradV=gradtheta1
    #             gradU1=gradphi1
    #             gradU2=gradphi2
    #             J=np.concatenate((-gradV.T, gradU1.T, -gradV.T, gradU2.T))
            
            
    #         udot=J@mm
    #         #print("udot= ", udot)
    #         u1=u+gamma*udot
    #         p1new=pos(u1[0],u1[1])
    #         p2new=pos(u1[2],u1[3])
    #         p1new=R.T@p1new
    #         p2new=R.T@p2new

    else:
        #In this case none of the surface normals are in z direction, hence no rotation matrix R is required, the angles are definite
        #print("case 2")
        phi1,theta1=calcAng(n1)
        phi2,theta2=calcAng(n2)
        gradphi1,gradtheta1=grad(phi1,theta1)
        gradphi2,gradtheta2=grad(phi2,theta2)
        J=np.zeros((4,3))

        u=np.array([[phi1],[theta1],[phi2],[theta2]])
        a=0
        if np.linalg.norm(gradphi1-gradphi2)<1e-4 :

            a=0
            u=np.array([[theta1],[phi1],[theta2],[phi2]])
            gradV=gradphi1
            gradU1=gradtheta1
            #print("grad U1= ",gradU1)
            gradU2=gradtheta2
            J=np.concatenate((-gradV.T, gradU1.T, -gradV.T, gradU2.T))

        elif np.linalg.norm(gradtheta1-gradtheta2)<1e-4:

            a=1
            u=np.array([[phi1],[theta1],[phi2],[theta2]])
            gradV=gradtheta1
            gradU1=gradphi1
            gradU2=gradphi2
            #print("grad U1= ",gradphi1)
            J=np.concatenate((-gradV.T, gradU1.T, -gradV.T, gradU2.T))
        
        
        #print("Jacobian J= ",J)
        udot=J@m
        #print("udot= ", udot)
        
        #print("u= ", u)
        u1=u+gamma*udot
        #print("u1 ", u1)
        #print("         a= ",a)
        if a==1:
            p1new=pos(u1[0],u1[1])
            p2new=pos(u1[2], u1[3])
        elif a==0:
            p1new=pos(u1[1],u1[0])
            p2new=pos(u1[3],u1[2])
    

    p1new=p1new+o1 #calculates p1 in local cube coordinates
    
    #print("p1 raw: ", p1new)
    p1new=proj1(p1new)
    p2new=p2new+o2
    
    #print("p2 raw: ", p2new.T)
    p2new=proj1(p2new)
    return p1new,p2new

def Force(p1: np.ndarray, p2: np.ndarray):

    #conducts one iteration of force residual control
    
    o1=getorigin1(p1)
    o2=getorigin1(p2)
    
    n1=getn(p1)
    n2=getn(p2)
    s=checksingularity(p1,n1,p2,n2)
    if s==1:
        print("singularity configuration detected")
    f=getF(p1,p2)
    gamma=-0.008
    if np.linalg.norm(np.absolute(n1)-np.array([[0],[0],[1]]))<1e-4:
        phi2,theta2 = calcAng(n2)
        gradphi2,gradtheta2=grad(phi2,theta2)
        phi1=phi2
        if np.linalg.norm(n1[2].item()-1)<1e-4: # if n1 is in +z direction theta1 needs to be 0, if its in- z direction then theta1 needs to be pi
            theta1=0
        else: 
            theta1=np.pi
        gradphi1,gradtheta1=grad(phi1,theta1)
        a=0
        u=np.array([[theta1],[phi1],[theta2],[phi2]])
        gradV=gradphi2
        gradU1=gradtheta1
        #print("grad U1= ",gradU1)
        gradU2=gradtheta2

    elif np.linalg.norm(np.absolute(n2)-np.array([[0],[0],[1]]))<1e-4:
        phi1,theta1=calcAng(n1)

        if np.linalg.norm(n2[2].item()-1)<1e-4:
            theta2=0
        else:
            theta2=np.pi
        
        gradphi1,gradtheta1=grad(phi1,theta1)
        phi2=phi1
        gradphi2,gradtheta2=grad(phi2,theta2)
        a=0
        u=np.array([[theta1],[phi1],[theta2],[phi2]])
        gradV=gradphi1
        gradU1=gradtheta1
        gradU2=gradtheta2
    
    else: #If none nof the contact normals points in z direction we simply calculate angles and gradients before comparing them and choosing correct u and v
        phi1,theta1=calcAng(n1)
        phi2,theta2=calcAng(n2)
        gradphi1,gradtheta1=grad(phi1,theta1)
        gradphi2,gradtheta2=grad(phi2,theta2)
        J=np.zeros((4,3))
        u=np.array([[phi1],[theta1],[phi2],[theta2]])

        if np.linalg.norm(gradphi1-gradphi2)<1e-4 :

            a=0
            u=np.array([[theta1],[phi1],[theta2],[phi2]])
            gradV=gradphi1
            gradU1=gradtheta1
            gradU2=gradtheta2

        elif np.linalg.norm(gradtheta1-gradtheta2)<1e-4:

            a=1
            u=np.array([[phi1],[theta1],[phi2],[theta2]])
            gradV=gradtheta1
            gradU1=gradphi1
            gradU2=gradphi2
    
    J=np.concatenate((gradU1.T, gradV.T, gradU2.T, gradV.T))
    #print(J)
    
    udot=J@f
    u1=u+gamma*udot
    #print(u1)
    if a==1:
        p1new=pos(u1[0],u1[1])
        p2new=pos(u1[2], u1[3])
            
    elif a==0: # In this case the grasp is in a singular (alternating) configuration. To avoid it we only move one finger while the other remains where it is
        p1new=pos(u1[1],u1[0])
        p2new=pos(u1[3],u1[2])

    p1new=p1new+o1
    #print("Force= ", f)
    #print("p1 raw: ", p1new)
    p1new=proj2(p1new)
    if s==1:
        p2new=p2
    
    else:
        p2new=p2new+o2
    
    #print("p2 raw: ", p2new)
        p2new=proj2(p2new)

    return p1new,p2new

def residualCtrl(p1: np.ndarray, p2: np.ndarray):
    #Implements switching grasp control by first running force residual control until a tolerance is achieved (Tolerance in this case is 1e-3) before continuing with moment residual control
    pcurr1=p1
    pcurr2=p2
    i=0
    while np.linalg.norm(getF(pcurr1,pcurr2))>1e-3:
        #print("iterating")
        pcurr1,pcurr2=Force(pcurr1,pcurr2)
        i=i+1
        if i>200:
            print("max iterations reached, something probabbly went wrong")
            break

    # print("rightF after Force Ctrl: ", pcurr1)
    # print("leftF after Force Ctrl: ", pcurr2)

    while np.linalg.norm(getM(pcurr1,pcurr2))>1e-3:
        pcurr1,pcurr2=Moment(pcurr1,pcurr2)

    return pcurr1,pcurr2

def Rot():
    #calculates Rotation matrix from global to local cube frame. The cube frame always has ez in global z direction
    #corner2,
    top,bottom,front,back,right,left = getWall(sim)
    com=sim.data.get_body_xpos("cube")
    ez=top-com
    ez=ez/np.linalg.norm(ez)
    ey=front-com
    ey=ey/np.linalg.norm(ey)
    ex=right-com
    ex=ex/np.linalg.norm(ex)
    R=np.array([ex, ey, ez])
    #R=R.T
    return R

def getWall(sim: mj.MjSim):
    #returns coordinates of all six cube wall centers in global coordinates
    center=sim.data.get_body_xpos('cube').reshape(-1,1)
    R=sim.data.get_body_xmat('cube')
    xdir=np.array([ 0.025, 0, 0]).reshape(-1,1)
    ydir=np.array([ 0, 0.025, 0]).reshape(-1,1)
    zdir=np.array([ 0, 0, 0.025]).reshape(-1,1)
    wx=center-R@xdir #calculate global coord of all 6 wall centers
    wx=wx.T
    wx=wx.reshape(3,)
    wx2=center+R@xdir
    wx2=wx2.T
    wx2=wx2.reshape(3,)
    wy=center-R@ydir
    wy=wy.T
    wy=wy.reshape(3,)
    wy2=center+R@ydir
    wy2=wy2.T
    wy2=wy2.reshape(3,)
    wz=center-R@zdir
    wz=wz.T
    wz=wz.reshape(3,)
    wz2=center+R@zdir
    wz2=wz2.T
    wz2=wz2.reshape(3,)
    ##Calculate max and min x,y,z vals of all wall centers
    # maxZ=np.max([wx[2],wx2[2],wy[2],wy2[2],wz[2],wz2[2]])
    # minZ=np.min([wx[2],wx2[2],wy[2],wy2[2],wz[2],wz2[2]])
    # maxY=np.max([wx[1],wx2[1],wy[1],wy2[1],wz[1],wz2[1]])
    # minY=np.min([wx[1],wx2[1],wy[1],wy2[1],wz[1],wz2[1]])
    # maxX=np.max([wx[0],wx2[0],wy[0],wy2[0],wz[0],wz2[0]])
    # minX=np.min([wx[0],wx2[0],wy[0],wy2[0],wz[0],wz2[0]])
    M=np.array([wx, wx2, wy, wy2, wz, wz2]) #Vectors in rows
    top=M[np.argmax(M[:,2]),:] #take row with highest z val
    bottom=M[np.argmin(M[:,2]),:] #take row with lowest z val
    front=M[np.argmax(M[:,1]),:] #take row with highest y val
    back=M[np.argmin(M[:,1]),:] #take row with lowest y val
    right=M[np.argmax(M[:,0]),:] #take row with highest x val
    left=M[np.argmin(M[:,0]),:] #take row with lowest x val
    cornerdir=np.array([ -0.025, -0.025, -0.025]).reshape(-1,1)

    #corner=center+R@cornerdir
    #corner=corner.T
    #corner=corner.reshape(3,)

    # c1=np.array([maxX, maxY, maxZ]) #old method for calculating corners, however these have many numerical inaccuracies, hence the function get corner is used instead
    # c2=np.array([maxX, maxY, minZ])
    # c3=np.array([maxX, minY, maxZ])
    # c4=np.array([maxX, minY, minZ])
    # c5=np.array([minX, maxY, maxZ])
    # c6=np.array([minX, maxY, minZ])
    # c7=np.array([minX, minY, maxZ])
    # c8=np.array([minX, minY, minZ])
    return top,bottom,front,back,right,left

def getCorner(sim: mj.MjSim):
    #returns coordinates of the most bottom left cube corner in global coordinates, used as origin for local cube coordinates
    center=sim.data.get_body_xpos('cube').reshape(-1,1)
    R=sim.data.get_body_xmat('cube') #Rotation matrix for transformation from local to global frame
    xdir=np.array([ 0.025, -0.025, -0.025]).reshape(-1,1)
    ydir=np.array([ -0.025, 0.025, -0.025]).reshape(-1,1)
    zdir=np.array([ -0.025, -0.025, 0.025]).reshape(-1,1)
    pdir=np.array([ -0.025, -0.025, -0.025]).reshape(-1,1)

    #In the following section the distance vector between COM and the corner points in cube COM coordinates are calculated

    wx=-R@xdir
    wxp=center+wx #calculate global coord of wall center
    wx=Rot()@wx #calculate cube corner in local cube coordinates with origin at com
    
    wx2=R@xdir
    wx2p=center+wx2
    wx2=Rot()@wx2
    
    wy=-R@ydir
    wyp=center+wy
    wy=Rot()@wy
    
    wy2=R@ydir
    wy2p=center+wy2
    wy2=Rot()@wy2
    
    wz=-R@zdir
    wzp=center+wz
    wz=Rot()@wz
    
    wz2=R@zdir
    wz2p=center+wz2
    wz2=Rot()@wz2
   
    ww=-R@pdir
    wwp=center+ww
    ww=Rot()@ww
    
    ww2=R@pdir
    ww2p=center+ww2
    ww2=Rot()@ww2
    
    A=np.array([wxp,wx2p,wyp,wy2p,wzp,wz2p,wwp,ww2p]) #array with global coord of cube corner points
    goal=np.array([[-0.025], [-0.025], [-0.025]]) #ideal distance vector between bottom left corner point and com in com reference frame
    #try to find which cornerpoint is most likely to match the goal
    nwx=np.linalg.norm(goal-wx)
    nwx2=np.linalg.norm(goal-wx2)
    nwy=np.linalg.norm(goal-wy)
    nwy2=np.linalg.norm(goal-wy2)
    nwz=np.linalg.norm(goal-wz)
    nwz2=np.linalg.norm(goal-wz2)
    nww=np.linalg.norm(goal-ww)
    nww2=np.linalg.norm(goal-ww2)
    i=np.argmin([nwx,nwx2,nwy,nwy2,nwz,nwz2,nww,nww2])
    #print(np.min([nwx,nwx2,nwy,nwy2,nwz,nwz2,nww,nww2]))
    corner=A[i].reshape(3,)
    # This calculation of corner doesn't work because it considers quaternion of cube as well resulting in odd gripperconfigurations:
    #corner=sim.data.get_body_xpos('cube').reshape(-1,1)+Rot()@R@goal; corner=corner.reshape(-1)

    return corner

def getcontactposV1(sim: mj.MjSim):
    #returns positions of finger link COMs
    #Note that these positions do not always equal to contact point position. However when sensor reading fails this is the best alternative
    #The positions are later projected onto the cube surface to get an actual contact point
    posright=sim.data.get_body_xpos("robot0:r_gripper_finger_link")
    posleft=sim.data.get_body_xpos("robot0:l_gripper_finger_link")
    
    print("pos right real: ", posright.T)

    #corner2,
    top,bottom,front,back,right,left=getWall(sim)
    corner=getCorner(sim)
    posright=Rot()@(posright-corner)
    posleft=Rot()@(posleft-corner)

    com=sim.data.get_body_xpos("cube")

    posright=posright.reshape(1,-1)
    posright=posright.T
    posleft=posleft.reshape(1,-1)
    posleft=posleft.T

    return posright,posleft

def getcontactposV2(sim: mj.MjSim):
    #returns positions of contacts, which are read from touch sensor
    # Note that this sensor does not alway work, because it sometimes does not register contact and returns NaN instead
    counterr=0
    counterl=0
    pr=np.array([0.0, 0.0, 0.0])
    pl=np.array([0.0, 0.0, 0.0])
    for i in range(sim.data.ncon):
        # Note that the contact array has more than `ncon` entries,
        # so be careful to only read the valid entries.
        contact = sim.data.contact[i]
        if sim.model.geom_id2name(contact.geom1) == "robot0:r_gripper_finger_link" :
            counterr=counterr+1
            pr+=contact.pos

        if sim.model.geom_id2name(contact.geom1) == "robot0:l_gripper_finger_link" :
            counterl=counterl+1
            pl+=contact.pos

    posright=pr/counterr
    posleft=pl/counterl
    #print("pos right real: ", posright.T)
    #corner2,
    top,bottom,front,back,right,left=getWall(sim)
    corner=getCorner(sim)

    posright=Rot()@(posright-corner)
    posleft=Rot()@(posleft-corner)

    posright=posright.reshape(1,-1)
    posright=posright.T
    posleft=posleft.reshape(1,-1)
    posleft=posleft.T
    return posright,posleft

def getcontactpos(sim: mj.MjSim):

    # returns contact positions from touch sensors via function getcontactposV2
    # If the sensors return NaN it gets the contact positions through finger links via getcontactposV1 instead

    #prV2,plV2=getcontactposV2(sim)
    #prV1,plV1=getcontactposV1(sim)
    # print("V2 right: ", prV2.T)
    # print("V2 left: ", plV2.T)
    # print("V1 right: ", prV1.T)
    # print("V1 left: ", plV1.T)
    pr,pl=getcontactposV2(sim)
    if np.isnan(pr).any() or np.isnan(pl).any():
        pr,pl=getcontactposV1(sim)
    return pr,pl


def setv():
    #function that returns arrays containing 9 test states with their corresponding mocap position and quaternion
    statearr=[]
    pos=[]
    quat=[]
    statearr.append(value(time=0.1200000000000008, qpos=np.array([0.43287507, 0.0349247, 0.68553313, 0.16433676, 0.91085971, 0.37713246, -0.03315303,  0.027,  0.02, 0.44999945, -0.04806348, 0.46847686, -0.00318028, -0.03297119, 0.00564472, 0.9994353]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
        0.0,  0.0, 0.0,  0.0, 0.0,
            0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    pos.append(np.array([0.4460014, -0.03224488, 0.4904613]))
    quat.append(np.array([0.71993394, -0.1160942, 0.67426239, -0.11656536]))

    statearr.append(value(time=0.1200000000000008, qpos=np.array([0.49677881, 0.15337997, 0.64254191, 0.1191527, 0.93434499, 0.33551392, -0.01524762,  0.05,  1e-5,  0.53376686, 0.12308308, 0.44573355, 0.73575101, -0.00629219, -0.04761504, 0.67554694]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
        0.0,  0.0, 0.0,  0.0, 0.0,
            0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    pos.append(np.array([0.50299078, 0.11808877, 0.39627194]))
    quat.append(np.array([0.7237781, -0.04759292, 0.67863408, -0.11548148])) 

    statearr.append(value(time=0.1200000000000008, qpos=np.array([0.41844007, 0.12766191, 0.68755642, 0.15425368, 0.92605884, 0.34419333, -0.01232015,  0.027, 0.022,  0.44056982, 0.06897918, 0.47093906, 0.72737217, -0.07362795, 0.00143882, 0.68228043]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
        0.0,  0.0, 0.0,  0.0, 0.0,
            0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    pos.append(np.array([0.43567277, 0.06673832, 0.49075421]))
    quat.append(np.array([0.7318267, -0.07823673, 0.66439189, -0.12996964])) 

    statearr.append(value(time=0.1200000000000008, qpos=np.array([0.55384916, 0.19707438, 0.63885158, 0.15298911, 0.92377688, 0.35060693, -0.01747514,  0.05, 1e-5, 0.59436152, 0.11555699, 0.43246026, 0.71318363, 0.0500337, -0.1189828, 0.68899117]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
        0.0,  0.0, 0.0,  0.0, 0.0,
            0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    pos.append(np.array([ 0.56486253, 0.1484074, 0.39591876]))
    quat.append(np.array([0.72875026, -0.08279316, 0.66809467, -0.12537091])) 

    statearr.append(value(time=0.1200000000000008, qpos=np.array([0.3623822, -0.01635806, 0.6737033, 0.14542997, 0.91855435, 0.36735762, -0.01250662, 0.027, 0.022, 0.38416018, -0.08110653,  0.44938658, 0.71249924, -0.07025653, 0.00656668, 0.69811584]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
        0.0,  0.0, 0.0,  0.0, 0.0,
            0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    pos.append(np.array([0.37971031, -0.07359567, 0.47581554]))
    quat.append(np.array([0.73068055, -0.08968511, 0.66827736, -0.10708818])) 

    statearr.append(value(time=0.1200000000000008, qpos=np.array([0.56959594, 0.15333703, 0.62989633, 0.19938805, 0.8928529, 0.39722741, -0.07258429, 0.037, 0.012, 0.58850584, 0.0575091, 0.426907, 0.99900783, -0.0355218, -0.00430843, 0.02651416]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
        0.0,  0.0, 0.0,  0.0, 0.0,
            0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    pos.append(np.array([0.57634779, 0.04872593, 0.39166058]))
    quat.append(np.array([0.69870544, -0.16867944, 0.68246025, -0.13268746])) 

    statearr.append(value(time=0.1200000000000008, qpos=np.array([0.55643788, 0.08466651, 0.65052159, 0.24416352, 0.89739378, 0.36717014, 0.01595842, 0.025, 0.024, 0.60254879, 0.01673821, 0.45023014, 0.07602249, -0.07557949, 0.84160583, 0.52934672]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
        0.0,  0.0, 0.0,  0.0, 0.0,
            0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    pos.append(np.array([0.59945785, -0.00363321, 0.46859221]))
    quat.append(np.array([0.76212993, -0.1520793, 0.60908084, -0.15827311])) 

    statearr.append(value(time=0.1200000000000008, qpos=np.array([0.52884015, 0.20799695, 0.62444155, 0.22391737, 0.88166558, 0.40474113, -0.09333508, 0.044, 0.02227, 0.54366171, 0.10070864, 0.4265464, 0.48859743, -0.00171332, 0.02258312, 0.87221535]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
        0.0,  0.0, 0.0,  0.0, 0.0,
            0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    pos.append(np.array([0.53138713, 0.10277217, 0.39216221]))
    quat.append(np.array([0.68129868, -0.19771884, 0.68958791, -0.14562927])) 

    statearr.append(value(time=1.420000000000001, qpos=np.array([ 0.43136836,  0.11711412,  0.41214772,  0.56099067,  0.70099295, 0.0928358 , -0.43044148,  0.01561888,  0.02731685,  0.29825713, -0.10553688,  0.42203406,  0.99818887, -0.03368179,  0.00369137,
        0.0497081 ]), qvel=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), act=None, udd_state={}))
    pos.append(np.array([0.35, -0.04, 0.38]))
    quat.append(np.array([ 0.35355339, -0.35355339,  0.61237244, -0.61237244]))

    return statearr, pos, quat

def cub2glob(p: np.ndarray):
    #Transforms input vector p from local frame to global frame
    corner=getCorner(sim).reshape(-1,1)
    
    glob=Rot().T@p + corner
    return glob

def rotx(phi: float):
    # returns rotation matrix around x axis for a rotation of phi degrees
    phi=2*np.pi*(phi/360)
    R=np.array([[1, 0, 0], [0, np.cos(phi),  -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    return R

def roty(theta: float):
    #returns Rotationmatrix for a rotation of theta around the y-axis
    #this function is used for a special case in the moment residual control algorithm
    theta=2*np.pi*(theta/360)
    R=np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return R

def rotz(phi: float):
    #returns Rotationmatrix for a rotation of phi around the x-axis
    phi=2*np.pi*(phi/360)
    R=np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    return R

def lr(phi: float): #Matrix function that describes all rotation matrices for mocap (gripper configurations) which can be used to grasp the leftside of the cube with the left finger and the rightside of the cube with the right finger
    lr=Rot().T@roty(90)@rotz(phi)
    return lr

def rl(phi: float): #Matrix function that describes all rotation matrices for mocap (gripper configurations) which can be used to grasp the leftside of the cube with the right finger and the rightside of the cube with the left finger
    rl=Rot().T@roty(90)@rotx(180)@rotz(phi)
    return rl

def bf(phi: float):
    bf=Rot().T@roty(90)@rotx(90)@rotz(phi)
    return bf

def fb(phi: float):
    fb=Rot().T@roty(90)@rotx(270)@rotz(phi)
    return fb

def tb(phi: float):
    tb=Rot().T@rotz(phi)
    return tb

def bt(phi: float):
    bt=Rot().T@rotx(180)@rotz(phi)
    return bt

def setquat(func , quat: np.ndarray):
    # inputs: matrix function (lr, rl,bf, fb, tb or bt) and quaternion of cube
    # Tries to find the function value (a rotation matrix) that has the smallest distance to the cube quaternion
    min=0
    dist=10
    for i in range(360):
        curr=mat2quat(func(i))
        #print("dist: ", i, " ", np.linalg.norm(curr-quat))
        if np.linalg.norm(curr-quat)<dist:
            min=i 
            #print("min= ", min)
            dist=np.linalg.norm(curr-quat) #set distance to next smallest value
    #print("min= ", min)
    sim.data.mocap_quat[0][:]= mat2quat(func(min))
    return func(min)

def check(p1: np.ndarray, p2: np.ndarray):

    #checks which gripper configuration (referring to quaternion of mocap) is needed for given goal points (depends on contact surface normals)
    #returns optimal gripper Rotation matrix for grasp improvement
    n1=getn(p1)
    n2=getn(p2)

    if n1[0]==1 and n2[0]==-1:
        q=setquat(lr,sim.data.mocap_quat[0][:]) #lr = leftfinger left, rightfinger right
    elif n1[0]==-1 and n2[0]==1:
        q=setquat(rl,sim.data.mocap_quat[0][:]) #rl = leftfinger right, rightfinger left 
    elif n1[1]==1 and n2[1]==-1:
        q=setquat(fb,sim.data.mocap_quat[0][:]) #fb = leftfinger back, rightfinger front
    elif n1[1]==-1 and n2[1]==1:
        q=setquat(bf,sim.data.mocap_quat[0][:]) #bf = leftfinger front, rightfinger back 
    elif n1[2]==1 and n2[2]==-1:
        q=setquat(tb,sim.data.mocap_quat[0][:]) #tb = leftfinger bottom, rightfinger top
    elif n1[2]==-1 and n2[2]==1:
        q=setquat(bt,sim.data.mocap_quat[0][:]) #bt = leftfinger top, rightfinger bottom 
    
    return q

def setpos(sim: mj.MjSim, view: mj.MjViewerBasic, g: np.ndarray):

    #Input g is the goal position for the right finger link 
    R= quat2mat(sim.data.mocap_quat[0][:])
    d=np.array([-0.05699, 0.00011, -0.02637]) #-0.0637 #relative distance between right finger link COM and Mocap when fingers have qpos of 0.035
    # This constant distance vector was calculated by transforming the distance between right finger and mocap from global to local coordinates in mocap frame
    #T=quat2mat(sim.data.mocap_quat[0][:])
    #dist=(sim.data.mocap_pos[0][:]-sim.data.get_body_xpos('robot0:r_gripper_finger_link')).reshape(-1,1)
    #d = T.T@dist
    d=d.reshape(-1,1)
    mpos=R@d +g #calculate required mocap position for desired
    mpos=mpos.reshape(3,)
    sim.data.mocap_pos[0][:]= mpos
    #print("r finger pos")
    #print(sim.data.get_body_xpos('robot0:r_gripper_finger_link'))

def setpos2(sim: mj.MjSim, view: mj.MjViewerBasic):#, g: np.ndarray):

    #Test function
    # Not relevant in any way

    #print(sim.data.get_body_xpos('panda_link7'))
    a=sim.data.get_joint_qpos('panda_joint7')
    pos=np.array([0.5, 0.2, 0.95])#a[0:3]
    quat=a[3:7]
    v=np.concatenate((pos,quat))
    sim.data.set_joint_qpos('panda_joint7', v)
    d=np.array([0.00704219, -0.03501198, -0.1575565])
    sim.data.mocap_pos[0][:]= pos+d
    sim.data.set_joint_qpos('robot0:r_gripper_finger_joint',0.05)
    noth(sim, mj_viewer, 125)
    print("dist for 0.05")
    print(sim.data.get_body_xpos('panda_link7')-sim.data.get_body_xpos('robot0:r_gripper_finger_link'))
    
    # print("before")
    # print(sim.data.mocap_pos[0][:])
    # print(sim.data.get_body_xpos('panda_joint7'))
    # print("dist:", sim.data.mocap_pos[0][:]-sim.data.get_body_xpos('panda_joint7'))

def setpos3(sim: mj.MjSim, view: mj.MjViewerBasic, g: np.ndarray):
    #similar to setps only this time it calculates the required link7 position which is necessary for desired mocap position g
    #this process accelerates the simulation because we do not have to wait for the gripper to reach the goal points

    R= quat2mat(sim.data.mocap_quat[0][:])
    d=np.array([-0.2069, 1.13e-4, -2.06e-2]) #z:-0.0637 or 0.0648 or -0.0566
    # d is relative distance between link 7 and right finger in local mocap frame
    #T=quat2mat(sim.data.mocap_quat[0][:])
    #dist=(sim.data.get_body_xpos('panda_link7')-sim.data.get_body_xpos('robot0:r_gripper_finger_link')).reshape(-1,1)
    #d = T.T@dist
    d=d.reshape(-1,1)
    link7pos=R@d +g #calculate optimal global position for link 7

    
    #print(link7pos)
    link7pos=link7pos.reshape(3,)
    return link7pos

# def rotate(sim: mj.MjSim, view: mj.MjViewerBasic, p1: np.ndarray, p2:np.ndarray):
#     dist=p1-p2
#     d2=sim.data.get_body_xpos('robot0:r_gripper_finger_link')-sim.data.get_body_xpos('robot0:l_gripper_finger_link')
#     while np.linalg.norm(d2-dist)>2e-3:

#             sim.data.mocap_quat[0][:]+=np.array([0, 0, 0, 0.3])

def correctGrasp(sim: mj.MjSim, view: mj.MjViewerBasic, prbefore: np.ndarray, prafter:np.ndarray, plbefore: np.ndarray, plafter: np.ndarray):

    # Inputs: Current and optimal contact positions for left and right finger

    # If current and optimal points are very close to one another no grasp correction is necessary
    # If that is not the case then

    if np.linalg.norm(prafter-prbefore)<1e-3 and np.linalg.norm(plafter-plbefore)<1e-3:
        print("Current finger positions are fine, no need for adjustment")

    else: 
        print("adjusting grasp")

        aa=sim.data.get_joint_qpos('cube:joint')
        a=np.copy(aa)
        oldQ=quat2mat(sim.data.mocap_quat[0][:]) #extracting rotation matrix of cube
        q=check(prafter,plafter)
        relq=relRot(sim, oldQ,q) #find link 7 quaternion for which gripper reaches desided grasping quaternion
        sim.data.mocap_pos[0][:]+=np.array([0.0, 0.0, 0.12])
        sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.035)
        sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.035)

        #noth(sim, mj_viewer, 185)
        #sim.data.set_joint_qpos('cube:joint',a)

        corner=getCorner(sim).reshape(-1,1)
        qq=Rot().T@prafter +corner #calculates global coordinates of right finger contact position
        setpos(sim, mj_viewer,qq) # mocap position set such that right finger pos is goal pos
        qq3=setpos3(sim, mj_viewer, qq) # Calculates link7 global position for which right finger reaches its desired position
        v=np.concatenate((qq3,relq))
        sim.data.set_joint_qpos('panda_joint7', v) # sets link7 to desired pose
        noth(sim, mj_viewer, 105)
        close(sim, mj_viewer)
        noth(sim, mj_viewer, 55)

def relRot(sim: mj.MjSim, oldR: np.ndarray, newR: np.ndarray):
    # NewR : Rotation matrix for transformation from global to desired mocap frame
    # oldR : Rotation matrix for transformation from global to current mocap frame
 
    relq=mat2quat(newR@oldR.T@sim.data.get_body_xmat('panda_link7')) 
    #oldR.T@xmat('panda_link7') defines the relative rotation from old mocap frame to link 7 frame
    #relq is the corresponding quaternion to the rotation matrix that transforms from global frame to link7 frame
    return relq

def close(sim: mj.MjSim, view: mj.MjViewerBasic):
    # closes Fingers until contact with object is registered
    pr,pl=getcontactposV2(sim)
    a=1
    b=1
    while a==1 or b==1 :
        if np.isnan(pr).any():
            sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')-0.0003)
        else:
            a=0
        
        if np.isnan(pl).any() :
            sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')-0.0003)
        else:
            b=0

        if sim.data.get_joint_qpos('robot0:r_gripper_finger_joint') < 0.0005:
            print("Failed closing right finger")
            break
        
        if sim.data.get_joint_qpos('robot0:r_gripper_finger_joint') < 0.0005:
            print("Failed closing left finger")
            break

        noth(sim, mj_viewer, 2)
        pr,pl=getcontactposV2(sim)


def start(sim: mj.MjSim, view: mj.MjViewerBasic, k: int):
    arr,poss,quat = setv()
    sim.set_state(arr[k])
    setp(poss[k], quat[k])
    #Gripper holding ob object oftentimes moved downwards after setting pose -->increased z value of mocap such that it stays static
    sim.data.mocap_pos[0][:]+= np.array([0.0, 0.0, 0.076])
    #on some occasions the fingers of the parallel jaw move apart
    #Therefor making them a tiny bit closer to the other makes it easier
    if sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')+sim.data.get_joint_qpos('robot0:l_gripper_finger_joint') >0.05:
        sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')-0.0005)
        sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')-0.0005)
    noth(sim, mj_viewer, 45)

xml_path = "/home/kaust/shortdextgen/envs/assets/PJ/flyingGr.xml"
#print(xml_path)
model = mj.load_model_from_path(xml_path)
sim = mj.MjSim(model)
try:
    #mj_viewer = mj.MjViewer(sim)#
    mj_viewer = mj.MjViewerBasic(sim)
    display_available = True
except mj.cymj.GlfwError:
    display_available = False

# sim.data.set_joint_qpos('cube:joint', [0.52959115, 0.13605998, 0.44289119, 0.73630197, -0.01732182, -0.83455705, 0.97554805])
# noth(sim, mj_viewer, 47)
# R=sim.data.get_body_xmat('cube')
# goal=np.array([[-0.025], [-0.025], [-0.025]])
# p=sim.data.get_body_xpos('cube').reshape(-1,1)+Rot()@R@goal
# #print(sim.data.get_body_xpos('cube'))
# p=p.reshape(-1)
# print(p)
# print(sim.data.get_body_xpos('cube')-getCorner(sim))
# print(getCorner(sim))
#print(getn(p))
#p11=np.array([[0.035],[0.035],[0.0]])
#p22=np.array([[0.025], [0.025], [0.05]])
# p1=np.array([[0.039],[0.025],[0.05]])
# p2=np.array([[0.05],[0.025],[0.025]])
# print(getM(p1,p2))
# s1,s2=Moment(p1,p2)
# s1,s2=Moment(s1,s2)
# s1,s2=Moment(s1,s2)
# s1,s2=Moment(s1,s2)
# pf1=np.array([[0.0499],[0.042],[0.05]])#np.array([[0.03],[0.04],[0.035]])
# pf2=np.array([[0.0],[0.0484],[0.0028]])#np.array([[0.03],[0.025],[-0.01]])
# #print(getn(pf2))
# s1,s2=Force(pf1,pf2)
# print(s1.T)
# print(s2.T)
# print(getM(s1,s2))
##begin
sim.data.mocap_quat[0][:]=np.array([0.99, 0.4 , 0.1, 0.8])
T=quat2mat(sim.data.mocap_quat[0][:])#sim.data.get_body_xmat('robot0:r_gripper_finger_link')
sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.035)
sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.035)
noth(sim, mj_viewer, 555)
#dist=sim.data.mocap_pos[0][:]-sim.data.get_body_xpos('robot0:r_gripper_finger_link')#-sim.data.mocap_pos[0][:]
dist=sim.data.get_body_xpos('panda_link7')-sim.data.get_body_xpos('robot0:r_gripper_finger_link')
dist=dist.reshape(-1,1)
print("mocap dist: ", T.T@dist)
#setpos2(sim, mj_viewer)
# start(sim, mj_viewer, 8)
# #noth(sim, mj_viewer, 485)
# goal=np.array([[-0.025], [-0.025], [-0.025]])
# #p=sim.data.get_body_xpos('cube').reshape(-1,1)+Rot()@sim.data.get_body_xmat('cube')@goal
# #print(p.reshape(-1))
# #print(getCorner(sim))
# pr,pl=getcontactpos(sim)
# prproj=proj2(proj3(pr))
# plproj=proj2(proj3(pl))
# print(prproj);print(plproj)
# prfinal,plfinal=residualCtrl(prproj,plproj)
# if np.linalg.norm(prfinal-prproj)<1e-3 and np.linalg.norm(plfinal-plproj)<1e-3:
#     print("left finger position fine")
# else:
#     correctGrasp(sim, mj_viewer,pr, prfinal, pl, plfinal)

# print("right:", prproj.T, "left:", plproj.T)
# print("right:", prfinal.T, "left:", plfinal.T)
# pr2,pl2=getcontactpos(sim)
# print(pr2.T)
#print("pos right calculated: ",cub2glob(pr).T)

# #qq=sim.data.get_joint_qpos('cube:joint')
# # print(sim.data.get_body_xpos('panda_link7'))
# # a=sim.data.get_joint_qpos('panda_joint7')
# # pos=a[0:3]
# # quat=a[3:7]
# #v=np.concatenate((pos,quat))
# #print("     v= ",v)
# #print("before")
# #print(sim.data.mocap_pos[0][:])

# #d=np.array([0.00704219, -0.03501198, -0.1575565])
# #print(sim.data.get_body_xpos('robot0:virtual_weld_link'))
# #print(sim.data.get_body_xpos('panda_link7'))
# #print(sim.data.mocap_pos[0][:]-sim.data.get_body_xpos('panda_link7'))
# #print(sim.data.get_body_xpos('panda_link7')-sim.data.get_body_xpos('robot0:r_gripper_finger_link'))
# #setpos(sim, mj_viewer)
# #sim.data.set_body_xpos('panda_link7')=np.array([0.5, 0.2, 0.8])
# #sim.set_state(value(time=0.1200000000000008, qpos=np.array([0.496, 0.0349247, 0.68553313, 0.16433676, 0.91085971, 0.37713246, -0.03315303,  0.027,  0.02, 0.44999945, -0.04806348, 0.46847686, -0.00318028, -0.03297119, 0.00564472, 0.9994353]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
#  #       0.0,  0.0, 0.0,  0.0, 0.0,
#   #          0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
# #sim.data.set_joint_qpos('cube:joint', qq)
# #sim.data.mocap_pos[0][:]+= np.array([0.1, 0.1, 0.06])
# #noth(sim, mj_viewer, 400)
# #print("after")
# #print(sim.data.mocap_pos[0][:]-sim.data.get_body_xpos('panda_link7'))
# #print(sim.data.mocap_pos[0][:]-sim.data.get_body_xpos('robot0:r_gripper_finger_link'))
# #print(np.isnan(pr).any())

# print(pr)
# print(proj2(proj3(pr)))
# prproj=proj2(proj3(pr))
# plproj=proj2(proj3(pl))
# #print("force: ", getF(prproj,plproj))
# print("right pos before: ",pr.T)

# print("right pos projected: ", prproj.T)
# prfinal,plfinal=residualCtrl(prproj,plproj)
# print("right pos optimized: ", prfinal.T)

# print(" ")
# if np.linalg.norm(prfinal-prproj)<1e-3:
#     print("right finger position is fine")
# else:
#     print("right finger pos needs adjustment")

# print(" ")

# print("left pos before: ",pl.T)

# print("left pos projected: ",plproj.T)
# print("left pos optimized: ", plfinal.T)



# if np.linalg.norm(plfinal-plproj)<1e-3:
#     print("left finger position fine")
# else:
#     print("left finger pos needs adjustment")

# aa=sim.data.get_joint_qpos('cube:joint')
# a=np.copy(aa)
# #print(a)
# rotz90=np.array([[0, -1, 0],[1, 0, 0], [0, 0, 1]])
# rotx90=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
# roty90=np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
# #sim.data.mocap_quat[0][:]= mat2quat(sim.data.get_body_xmat("cube")@roty90) #+=np.array([0, 0, 0, 0.3])
# #sim.data.mocap_pos[0][:]+= np.array([0.0, 0.0, 0.76])

# correctGrasp(sim, mj_viewer,pr, prfinal, pl, plfinal)


# ###end
# #a2=np.array([0.05,   0.02302306, 0.05860228])
# #print(proj2(a2))
# # glR=quat2mat(sim.data.mocap_quat[0][:])
# # q=check(prfinal,plfinal)#.reshape(4,)
# # #noth(sim, mj_viewer, 10)
# # relq=relRot(sim, glR,q)#mat2quat(q@glR.T@sim.data.get_body_xmat('panda_link7'))#relRot(sim,q)
# # #noth(sim, mj_viewer, 85)
# # corner=getCorner(sim).reshape(-1,1)
# # qq=Rot().T@prfinal +corner
# # setpos(sim, mj_viewer,qq)
# # #qq=qq.reshape(3,)# +np.array([0,0, 0.9])
# # qq2=sim.data.get_body_xpos('panda_link7')+np.array([0, 0, 0.2])
# # qq3=setpos3(sim, mj_viewer, qq)
# # #pos=np.array([0.5, 0.2, 0.95])#a[0:3]
# # #quat=a[3:7]#np.array([1, 0, 0, 0])#a[3:7]
# # v=np.concatenate((qq3,relq))
# # sim.data.set_joint_qpos('panda_joint7', v)

# # #sim.data.mocap_pos[0][:]+=np.array([0.0, 0.0, 0.012])
# # sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')+0.027)
# # sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')+0.027)

# # noth(sim, mj_viewer, 100)
# # #sim.data.mocap_pos[0][:]+=np.array([0.0, 0.0, -0.5])
# # #noth(sim, mj_viewer, 85)
# # sim.data.set_joint_qpos('cube:joint',a)#+np.array([0, 0, 0.7, 0, 0, 0, 0]))

# # noth(sim, mj_viewer, 5)
# # corner=getCorner(sim).reshape(-1,1)
# # qq=Rot().T@prfinal +corner
# # sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.05)
# # sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.05)
# # #noth(sim, mj_viewer, 85)
# # setpos(sim, mj_viewer,qq)
# # sim.data.set_joint_qpos('cube:joint',a)


# # #sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')-0.002)
# # #sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')-0.002)
# # #noth(sim, mj_viewer, 105)
# # close(sim, mj_viewer)
# pp,pp2=getcontactposV1(sim)
# print("Now: ")
# print("right f: ", proj2(proj3(pp)))
# print("left f ", proj2(proj3(pp2)))
#setpos(sim, mj_viewer,qq)
#noth(sim, mj_viewer, 45)
#print(Roty(270))
#p1final,p2final=Force(pf1,pf2)
#p1final=getn(pf1)

#p1final,p2final=residualCtrl(pf1,pf2)#Force(pf1,pf2)#Moment(p11,p22)

#p1final,p2final=Force(p1final,p2final)
# for i in range(5):
# #   p1final,p2final=Moment(p1final,p2final)
#     p1final,p2final=Force(p1final,p2final)

#print("p1: ")
#print(p1final)
#print("p2: ")
#print(p2final)