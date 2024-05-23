import mujoco_py as mj
import ipopt
import cyipopt
import numpy as np
import jax
from jax import jit, grad, jacfwd
import numdifftools as nd
import time
import pickle

class value(): #value class defined to set and copy mujoco states
    def __init__(self,time,qpos,qvel,act,udd_state):
        self.time=time
        self.qpos=qpos
        self.qvel=qvel
        self.act=act
        self.udd_state=udd_state

def noth(sim: mj.MjSim, view: mj.MjViewerBasic, num: int): # runs simulation for num steps
    for i in range(num):
        time.sleep(0.04)
        sim.step()
        mj_viewer.render()

def setp(a: np.ndarray, b: np.ndarray): # sets mocap to position a and quaternion b
    sim.data.mocap_pos[0][:]=a
    sim.data.mocap_quat[0][:]=b

def askGr():     # ask user to choose sample grasp index
    while True:
        index= input("Enter Grasp index between 0 and 7 ")
        try:
            index=int(index)
            if index >-1 and index<8:
                break
            else:
                print("Index must lie between 0 and 7, try again")
        except ValueError:
            print("Value must be a NUMBER")
    return index

def askVar():  #asks user to choose contact model and number of friction cone edges
    while True:
        model= input("Enter 0 for PCWF or 1 for SFC ")
        try:
            model=int(model)
            if model ==0 or model==1:
                break
            else:
                print("Number can't be other than 0 or 1, try again")
        except ValueError:
            print("Value must be a NUMBER")
    
    while True:
        num= input("Enter number of linearized Friction cone edges ")
        try:
            num=int(num)
            if num >2:
                break
            else:
                print("number has to be greater than 2, try again")
        except ValueError:
            print("Value must be a NUMBER")

    return model, num

def edgeV(num: int, mu: float): # returns a Matrix containing s friction cone edge vectors, corresponds to equation 2.8

    pi=np.pi
    F=[]
    for i in range(num):
        a=mu*np.cos((2*pi*i)/num)
        if np.abs(a)<1e-6:
            a=0
        b=mu*np.sin((2*pi*i)/num)
        if np.abs(b)<1e-6:
            b=0
        f=[1, a, b]
        F.append(f)
    F=np.array(F)
    F=F.T
    return F

def createF(nc: int, s: int, mu: float):  # returns the combined friction cone matrix F for all fingers as defined in equation 3.10 and 3.11
    F=edgeV(s, mu) 
    Fges=np.kron(np.eye(nc,dtype=float), F) # creates a block diagoonal matrix with nc times F
    
    return Fges

def getWall(sim: mj.MjSim): #returns coordinates of all six cube wall centers in global coordinates
    center=sim.data.get_body_xpos('cube').reshape(-1,1)
    R=sim.data.get_body_xmat('cube')
    #define relative distance vecors the wall centers can have with regard to the cube's com
    xdir=np.array([ 0.025, 0, 0]).reshape(-1,1)
    ydir=np.array([ 0, 0.025, 0]).reshape(-1,1)
    zdir=np.array([ 0, 0, 0.025]).reshape(-1,1)
    #calculate global coord of all 6 wall centers:
    wx=center-R@xdir
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

    M=np.array([wx, wx2, wy, wy2, wz, wz2]) #wall center vectors in rows
    # M[:,i] is the i th column of matrix M
    top=M[np.argmax(M[:,2]),:] #take row with highest z val
    bottom=M[np.argmin(M[:,2]),:] #take row with lowest z val
    front=M[np.argmax(M[:,1]),:] #take row with highest y val
    back=M[np.argmin(M[:,1]),:] #take row with lowest y val
    right=M[np.argmax(M[:,0]),:] #take row with highest x val
    left=M[np.argmin(M[:,0]),:] #take row with lowest x val

    return top,bottom,front,back,left,right

def Rot():  #calculates Rotation matrix from global to local cube frame. 
   
    #This Rotationmatrix differs from the classical cube com fram matrix as this one determines the current sides of the cube (top, bottom etc.) with regard to the current configuration which are not hardcoded. So for instance ez alway points towards the top most surface of the cube

    top,bottom,front,back,left,right = getWall(sim)
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

def getcontactposV2(sim: mj.MjSim):     # returns positions of contacts, which are read from touch sensor
    # Note that this sensor does not always work, it sometimes does not register contact with the object and returns NaN instead
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

    return posright,posleft

def getcontactpos(sim: mj.MjSim):  #returns finger contact positions either through sensor or finger link com
    pr,pl=getcontactposV2(sim)
    if np.isnan(pr).any() or np.isnan(pl).any():   #if sensor does not work use finger link com
        pr=sim.data.get_body_xpos("robot0:r_gripper_finger_link")
        pl=sim.data.get_body_xpos("robot0:l_gripper_finger_link")
    return pr,pl

def checkside(sim: mj.MjSim):                       #returns com R c_i, this function is specific for a cube, the rotation matrices defined here are also deficted in figure 3.1
    top,bottom,front,back,left,right = getWall(sim)
    W=np.array([top, bottom, front, back, left, right])
    p1,p2=getcontactpos(sim)
    A=np.array([])
    B=np.array([])
    for i in range(6):
        A=np.append(A, np.array([np.linalg.norm(p1-W[i,:])])) #calculates distance from contact point to each wall center
        B=np.append(B, np.array([np.linalg.norm(p2-W[i,:])]))
    if np.argmin(A)==0 and np.argmin(B)==1 : #if right contact point has the shortest distance from top and left one from bottom return these rotation matrices
        #print("right top, left bottom")
        R1=np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        R2=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    elif np.argmin(A)==1 and np.argmin(B)==0 : # Rotation matrices in case right finger at bottom side and left finger at top side
        #print("left top, right bottom")
        R2=np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        R1=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    elif np.argmin(A)==2 and np.argmin(B)==3 : # Rotation matrices in case right finger at front side and left finger at back side
        #print("right frontside, left backside")
        R1=np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        R2=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    elif np.argmin(A)==3 and np.argmin(B)==2 :  # Rotation matrices in case right finger at back side and left finger at front side
        #print("left frontside, right backside")
        R2=np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        R1=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    elif np.argmin(A)==4 and np.argmin(B)==5 :  # Rotation matrices in case right finger at left side and left finger at right side
        #print("right leftside, left rightside")
        R1=np.eye(3)
        R2=np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif np.argmin(A)==5 and np.argmin(B)==4 :  # Rotation matrices in case right finger at right side and left finger at left side
        #print("left leftside, right rightside")
        R2=np.eye(3)
        R1=np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        print("no valid grasping position")  # We require that the fingers are located on opposing sides,  a grasp with finger not on opposing sides would only work for extremely large friction coeffiecients or multiple fingers
        R1=np.zeros(3)
        R2=np.zeros(3)
    R1=R1.T
    R2=R2.T

    R=Rot() #calculate global transformation to the given surface frame
    R1=R1@R 
    R2=R2@R
    return R1,R2

def getFdir(sim: mj.MjSim): # returns the 3 dimensional contact force vector that each finger applies on the object in their corresponding surface frame
    m=sim.data.get_body_xmat("robot0:r_gripper_finger_link")
    n=sim.data.get_body_xmat("robot0:l_gripper_finger_link")
    R1,R2=checkside(sim)
    assert R1.shape==(3,3)
    assert R2.shape==(3,3)
    kr=-R1@m[:,1]  #removed -, since changes in xml file made it unnecessary
    kr=kr.reshape(-1,1)
    kl=R2@n[:,1]
    kl=kl.reshape(-1,1)

    return kr, kl


def GraspMatrix(sim: mj.MjSim, mode: int):  # return grasp matrix for current configuration, based on equations 2.14 and 2.22
    Rr,Rl= checkside(sim)
    cr,cl=getcontactpos(sim)
    com=sim.data.get_body_xpos("cube")
    ar=cr-com

    Sr=np.array([[0, -ar[2], ar[1]], [ar[2],0, -ar[0]], [-ar[1], ar[0], 0]])  # hat operator on distance vector between right finger and com
    Gr=np.block([[Rr, np.zeros(np.shape(Rr))],[Sr@Rr, Rr]])  # partial grasp matrix for right finger

    al=cl-com
    Sl=np.array([[0, -al[2], al[1]], [al[2],0, -al[0]], [-al[1], al[0], 0]])   # hat operator on distance vector between left finger and com
    Gl=np.block([[Rl, np.zeros(np.shape(Rl))],[Sl@Rl, Rl]])  # partial grasp matrix for left finger
    Gtilde=np.block([Gr, Gl])
    
    if mode ==0:
        #PCWF case
        B1=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    elif mode==1:
        #SFC case
        B1=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    B=np.block([[B1, np.zeros(np.shape(B1))], [np.zeros(np.shape(B1)), B1]])
    G=np.block([Gr@B1, Gl@B1]) # actual grasp matrix
    return G

def getQ(s: int, mode: int):  # returns matrix Q and vector q as defined in equation 3.15
    nc=2
    if mode==0:
        li=3
        
    if mode==1:
        li=4
    
    l=nc*li
    I=np.eye(l)
    O=np.zeros((nc*s,nc*s))
    Q=np.block([[I, np.zeros((l, nc*s))], [np.zeros((nc*s, l)), O]])
    q=np.block([np.ones(l), np.zeros(nc*s)])
    return Q,q

def getTW(s: int, mode: int): # returns matrices T and W as defined in equation 3.17 and 3.18
    nc=2
    if mode==0:
        li=3
    
    if mode==1:
        li=4

    l=nc*li
    I1=np.eye(l)
    I2=np.eye(nc*s)
    O1=np.zeros((l, nc*s))
    O2=np.zeros((nc*s, l))
    T=np.block([I1, O1])
    W=np.block([O2, I2])
    return T,W

def init(sim: mj.MjSim, s:int, mode:int):  # returns initial state vector x depending on contact model, initial force values are the unit force directions
    nc=2
    f1,f2=getFdir(sim)
    if mode==0:
        li=3
    
    if mode==1:
        li=4
        f1=np.concatenate((f1,np.zeros((1,1))))  # set initial contact torque to zero, append it to force vector
        f2=np.concatenate((f2,np.zeros((1,1))))
    x=np.concatenate((f1,f2))
    v=x
    v=v.reshape(-1,1)
    Fg=createF(nc, s, 0.25)

    a=np.ones(nc*s).reshape(-1,1) # we set the initial parameter values to 0

    x=np.concatenate((x,a))
    return x

def createBounds(s:int, mode:int):  # returns lower and upper bound vector
    nc=2
    if mode==0:
        li=3
    
    elif mode==1:
        li=4
    

    l=nc*li
    v=np.ones((l+nc*s))
    lb=-1e3*v
    ub=1e3*v
    for i in range(nc):
        lb[i*li]=-1e-6     # force in normal direction (In this case x direction) needs to be greater than 0
    lb[l:l+nc*s]=-1e-6      # the parameters alpha_ci need to be greater or equal to zero

    return lb,ub

def getConst(mode:int, s:int): # function that returns constraint vector for inequality constraints depending on the contact model

    # Equality constraints are approximated by giving them extremely small lower and upper bounds
    nc=2
    if mode==0:
        li=3
        l=nc*li
        v=np.ones((6+l))  # for PCWF we only have equality constraints (see 3.21) (3.19 and 3.20  excluded as those are constraints on the stae vector x itself) which is why we set the lower and upper bounds very low (close to zero)
        cl=-1e-6*v
        cu=1e-6*v
    elif mode==1:
        li=3
        l=nc*li
        v=np.ones((6+l+nc*2))
        cl=-1e-6*v
        cu=1e-6*v
        cu[6+l:6+l+2*nc]=1e6 # there is no upper bound for torsional friction inequality, only lower bound
    return cl,cu

def getN(nc: int):   # returns Filtermatrix N as defined in 3.24
    N=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])  # Nci for SFC as defined in 3.23
    Nges=np.kron(np.eye(nc,dtype=float), N)
    return Nges

class Quad(object):    # Clsass for our optimization problem which is solved with intopt, we assume that the force optimization occurs on a two fingered gripper while grasping a cube
    def __init__(self,num,mode):
        self.s=num
        self.nc=2
        self.mode=mode
        self.factor=1
        pass

    def objective(self, x): # objective function as defined in 3.13
        #
        # The callback for calculating the objective
        #
        Q,q=getQ(self.s, self.mode)
        return self.factor*0.5*x.T@Q@x + q@x

    def gradient(self, x):  # returns gradient of objective function
        #
        # The callback for calculating the gradient
        #
        Q,q=getQ(self.s, self.mode)
        return self.factor*Q@x + q.T

    def constraints(self, x): # returns constraint vector cg which has all constraints stacked into one vector
        #
        # The callback for calculating the constraints
        #
        G=GraspMatrix(sim, self.mode)
        mu=0.33
        T,W=getTW(self.s,self.mode)
        v=T@x # extracting contact force vector fc
        a=G@v
        checkside(sim)
        a[2]=a[2]-18
        #a corresponds to the force closure contstraint as defined in equation 2.29
        F=createF(self.nc,self.s, mu)
        
        cg=a
        if self.mode ==0:
            c1=T@x-F@W@x  #refers to the constraint defined in equation (3.10)
        
        elif self.mode==1: #SFC mode also has inequality constraints with torsional friction coefficient
            li=4
            N=getN(self.nc)
            c1=N@T@x-F@W@x  # Matrix N also used because we are in SFC case
            c=np.array([])
            gamma=0.2
            for i in range(self.nc):
                cc=gamma*x[li*i]-x[li*i+3]
                cc=np.array(cc)
                cc2=gamma*x[li*i]+x[li*i+3]  # cc and cc2 correspond to the inequality constraints define in 3.22
                cc2=np.array(cc2)
                c=np.append(c,cc)
                c=np.append(c,cc2)
                c=c.reshape(-1,1)
            c1=np.append(c1,c)
        cg=np.append(cg,c1)
        cg=cg.reshape(-1,1)
        #print("CG SHAPE= ", cg.shape)
        return cg

    def jacobian(self, x): #evaluates jacobian of constraints at x
        #
        # The callback for calculating the Jacobian
        #
        print(np.linalg.matrix_rank(nd.Jacobian(self.constraints)(x)))
        # print(nd.Jacobian(self.constraints)(x).shape)
        return nd.Jacobian(self.constraints)(x)

    def hessian(self, x, lagrange, obj_factor): #returns hessian of objective function
        #
        # The callback for calculating the Hessian
        #
        Q,q=getQ(self.s, self.mode)
        return self.factor*Q


def setv():     #function that returns arrays containing 9 test states with their corresponding mocap position and quaternion, These test sets were used to test the residual controllers
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

    # statearr.append(value(time=0.1200000000000008, qpos=np.array([0.48430023, 0.02597056, 0.64269951, 0.11160507, 0.93756286, 0.32733491, 0.03704109,  0.017, 0.012, 0.45697091, -0.11607787, 0.4246814, 0.46237869, -0.46327329, -0.53433676, 0.53485328]), qvel=np.array([ 0.0, 0.0,  0.0, 0.0,  0.0,
    #        0.0,  0.0, 0.0,  0.0, 0.0,
    #         0.0,  0.0,  0., 0.0]), act=None, udd_state={}))
    # pos.append(np.array([ 0.51179956, -0.00772376, 0.39264169]))
    # quat.append(np.array([0.7558128, -0.02306977, 0.6462006, -0.10314837]))  not a grasp

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


def runopt(sim: mj.MjSim, s:int, mode:int):         # runs optimization with s number of edges in linearized friction cone and mode which selects PCWF of SFC

    # lb, ub : lower and upper bounds for state vector x
    # cu, cl : lower and upper bounds for constraints of optimization problem

    #checkside(sim)
    x0=init(sim, s, mode)
    lb, ub= createBounds(s, mode)
    cl,cu=getConst(mode, s)
    solver=ipopt.problem(n=len(x0),
                m=len(cl),
                problem_obj=Quad(s, mode),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )
    solver.addOption('tol', 1e-9)
    solver.addOption('acceptable_tol', 1e-8)
    #solver.add_option('mu_strategy','adaptive')
    solver.addOption('max_iter', 490) # maximum iterations before aborting optimization
    x,info =solver.solve(x0)

    #Note that oldr and oldl have norm 1
    # We project the newly calculated force onto the old one such that we know the absolute force we need to apply on the initial contact direction
   
    if mode==0: #PCWF mode
        frr=x[0:3] #right finger force vector
        fr=frr/np.linalg.norm(frr)  #calculating new force direction
        fll=x[3:6] #left finger force vector
        fl=fll/np.linalg.norm(fll) #calculating new force direction
        oldr,oldl=getFdir(sim)

        projr=frr@oldr  #project new force onto old one 
        projl=fll@oldl 

    elif mode==1:

        frr=x[0:3]
        fr=frr/np.linalg.norm(frr)
        fll=x[4:7]
        fl=fll/np.linalg.norm(fll)
        oldr,oldl=getFdir(sim)
        projr=frr@oldr  #project new force onto old one
        projl=fll@oldl
    
    return projr,projl


xml_path = "/home/kaust/shortdextgen/envs/assets/PJ/flyingGrG.xml"
model = mj.load_model_from_path(xml_path)
sim = mj.MjSim(model)
try:
    #mj_viewer = mj.MjViewer(sim)#
    mj_viewer = mj.MjViewerBasic(sim)
    display_available = True
except mj.cymj.GlfwError:
    display_available = False

if __name__ == '__main__':

   


    

# Use following for force optimization on larger test set of grasps after position optimization
    # file1 = open('state-set', 'rb') #open('opt-states', 'rb')
    # states = pickle.load(file1)
    # file1.close()
    # file2 = open('pos-set', 'rb') #open('opt-positions', 'rb')
    # poss = pickle.load(file2)
    # file2.close()
    # file3 = open('quat-set', 'rb') #open('opt-quaternions', 'rb')
    # quat2 = pickle.load(file3)
    # file3.close()
    
    index=askGr()
    sim.set_state(statearr[index])
    setp(pos[index], quat[index])
    #alternative:
    #i between 0 and 82
    #sime.set_state(states[i])
    #setp(poss[i], quat2[i])

    noth(sim, mj_viewer, 20)
    mode, s= askVar()
    x0=init(sim, s, mode)
    print(x0)
    lb, ub= createBounds(s, mode)
    cl,cu=getConst(mode, s)
    #print(Quad(s,mode).constraints(x0))
    solver=ipopt.problem(n=len(x0),
                m=len(cl),
                problem_obj=Quad(s, mode),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )
    solver.addOption('tol', 1e-9)
    #solver.addOption('acceptable_iter', 10)
    solver.addOption('acceptable_tol', 1e-8)
    #solver.add_option('mu_strategy','adaptive')
    solver.addOption('max_iter', 1090)
    # x,info =solver.solve(x0)
    x,info =solver.solve(x0)
    print("x= ")
    print(x)
    print("Constraints evaluated at x: ")
    print(Quad(s,mode).constraints(x))


    
    print("successfully completed")
