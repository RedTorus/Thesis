import mujoco_py as mj
import ipopt
import cyipopt
import numpy as np
import jax
from jax import jit, grad, jacfwd
import numdifftools as nd
import time
import pickle

class value():
    def __init__(self,time,qpos,qvel,act,udd_state):
        self.time=time
        self.qpos=qpos
        self.qvel=qvel
        self.act=act
        self.udd_state=udd_state

def noth(sim: mj.MjSim, view: mj.MjViewerBasic, num: int):
    for i in range(num):
        time.sleep(0.04)
        #sim.data.mocap_pos[0][:]=np.array([ 0.01*i,  -0.0003224488*i,  0.004904613*i ])
        
        sim.step()
        mj_viewer.render()

def setp(a: np.ndarray, b: np.ndarray):
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
    #mu=0.2
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
        #print("edge vect nr: ", i, "= ", f)
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
    #Calculate max and min x,y,z vals of all wall centers
    maxZ=np.max([wx[2],wx2[2],wy[2],wy2[2],wz[2],wz2[2]])
    minZ=np.min([wx[2],wx2[2],wy[2],wy2[2],wz[2],wz2[2]])
    maxY=np.max([wx[1],wx2[1],wy[1],wy2[1],wz[1],wz2[1]])
    minY=np.min([wx[1],wx2[1],wy[1],wy2[1],wz[1],wz2[1]])
    maxX=np.max([wx[0],wx2[0],wy[0],wy2[0],wz[0],wz2[0]])
    minX=np.min([wx[0],wx2[0],wy[0],wy2[0],wz[0],wz2[0]])
    M=np.array([wx, wx2, wy, wy2, wz, wz2]) #Vectors in rows
    top=M[np.argmax(M[:,2]),:] #take row with highest z val
    bottom=M[np.argmin(M[:,2]),:]
    front=M[np.argmax(M[:,1]),:]
    back=M[np.argmin(M[:,1]),:]
    right=M[np.argmax(M[:,0]),:]
    left=M[np.argmin(M[:,0]),:]

    # c1=np.array([maxX, maxY, maxZ]) #corner
    # c2=np.array([maxX, maxY, minZ])
    # c3=np.array([maxX, minY, maxZ])
    # c4=np.array([maxX, minY, minZ])
    # c5=np.array([minX, maxY, maxZ])
    # c6=np.array([minX, maxY, minZ])
    # c7=np.array([minX, minY, maxZ])
    # c8=np.array([minX, minY, minZ])
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
    #prV2,plV2=getcontactposV2(sim)
    #prV1,plV1=getcontactposV1(sim)
    # print("V2 right: ", prV2.T)
    # print("V2 left: ", plV2.T)
    # print("V1 right: ", prV1.T)
    # print("V1 left: ", plV1.T)
    pr,pl=getcontactposV2(sim)
    if np.isnan(pr).any() or np.isnan(pl).any():   #if sensor does not work use finger link com
        pr=sim.data.get_body_xpos("robot0:r_gripper_finger_link")
        pl=sim.data.get_body_xpos("robot0:l_gripper_finger_link")
    return pr,pl

def checkside(sim: mj.MjSim):                       #returns com R c_i, this function is specific for a cube, the rotation matrices defined here are also deficted in figure 3.1
    top,bottom,front,back,left,right = getWall(sim)
    W=np.array([top, bottom, front, back, left, right])
    #p1=sim.data.get_body_xpos("robot0:r_gripper_finger_link")
    #p2=sim.data.get_body_xpos("robot0:l_gripper_finger_link")
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
    #rel1=R1.T
    #rel2=R2.T
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
    #kinR=kinR.T
    #cr=sim.data.get_body_xpos("robot0:r_gripper_finger_link")
    cr,cl=getcontactpos(sim)
    com=sim.data.get_body_xpos("cube")
    ar=cr-com

    Sr=np.array([[0, -ar[2], ar[1]], [ar[2],0, -ar[0]], [-ar[1], ar[0], 0]])  # hat operator on distance vector between right finger and com
    Gr=np.block([[Rr, np.zeros(np.shape(Rr))],[Sr@Rr, Rr]])  # partial grasp matrix for right finger
    #kinL=kinL.T
    #cl=sim.data.get_body_xpos("robot0:l_gripper_finger_link")
    al=cl-com
    Sl=np.array([[0, -al[2], al[1]], [al[2],0, -al[0]], [-al[1], al[0], 0]])   # hat operator on distance vector between left finger and com
    Gl=np.block([[Rl, np.zeros(np.shape(Rl))],[Sl@Rl, Rl]])  # partial grasp matrix for left finger
    Gtilde=np.block([Gr, Gl])
    
    if mode ==0:
        #PCWF cse
        B1=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    elif mode==1:
        #SFC case
        B1=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    B=np.block([[B1, np.zeros(np.shape(B1))], [np.zeros(np.shape(B1)), B1]])
    #Gt=B@Gtilde.T
    G=np.block([Gr@B1, Gl@B1])#Gtilde@B.T#Gt.T   # actual grasp matrix
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
        f1=np.concatenate((f1,np.zeros((1,1))))
        f2=np.concatenate((f2,np.zeros((1,1))))
    x=np.concatenate((f1,f2))
    v=x
    v=v.reshape(-1,1)
    Fg=createF(nc, s, 0.25)
    # if mode==1:
    #     N=getN(nc)
    #     v=N@v

    #a=np.linalg.pinv(Fg)@v

    #print(Fg.shape)
    #a=a.flatten()
    a=np.ones(nc*s).reshape(-1,1)#np.zeros(nc*s).reshape(-1,1)
    #print(a.shape)
    #a=a*1/s
    #a[0]=1
    # a[0]=0.5059;a[s]=0.5258
    # a[1]=0.0198;a[s+1]=-0.02
    # a[3]=0.4733;a[s+2]=0.4932
    # a[4]=0;a[s+3]=0
    x=np.concatenate((x,a))
    #x=x.reshape(-1,1)
    #x=100*x
    #x=x.T
    return x

def createBounds(s:int, mode:int):
    print("entered here")
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
        lb[i*li]=-1e-6
        #lb[l+i*s]=0
    lb[l:l+nc*s]=-1e-6

    return lb,ub

def getConst(mode:int, s:int):
    # function that returns constraint vector depending on the contact model
    nc=2
    if mode==0:
        li=3
        l=nc*li
        v=np.ones((6+l))
        cl=-1e-6*v
        cu=1e-6*v
    elif mode==1:
        li=3
        l=nc*li
        v=np.ones((6+l+nc*2))
        cl=-1e-6*v
        cu=1e-6*v
        cu[6+l:6+l+2*nc]=1e6 # entries with indices greater than 6+l have the value
        if s%2 ==1:
            
            cl[6+l:6+l+2*nc]=-1e6
        #print("CU shape= ", cu.shape)

    #cl = -np.ones(16)
    #cu = np.ones(16)

    return cl,cu

def getN(nc: int):   # returns Filtermatrix N as defined in 3.24
    N=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])  # Nci for SFC as defined in 3.23
    Nges=np.kron(np.eye(nc,dtype=float), N)
    return Nges

class Quad(object):
    def __init__(self,num,mode):
        self.s=num
        self.nc=2
        self.mode=mode
        self.factor=1
        #G=GraspM(R1, R2)
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
        G=GraspMatrix(sim, self.mode)#np.array([[1, 0, 0, -1, 0, 0],[0, 1, 0, 0, -1, 0], [0, 0, 1, 0, 0, 1],[0, 0, 0, 0, 0, 0],[0, 0, -1, 0, 0, 1],[0, 1, 0, 0, 1, 0]])
        mu=0.33
        T,W=getTW(self.s,self.mode)
        v=T@x # extracting contact force vector fc
        a=G@v
        checkside(sim)
        #print("VARIABLE: ")
        #print(a)
        #a[0:6]=0
        a[2]=a[2]-18
        #a corresponds to the force closure contstraint as defined in equation 2.29
        F=createF(self.nc,self.s, mu)
        
        cg=a
        #cg=np.array([a[2]])#np.array([x[2]+x[5]-20])#G@x
        #cg[2]=cg[2]-20
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
                c=np.append(c,cc)#np.concatenate((c,cc, cc2))
                c=np.append(c,cc2)
                c=c.reshape(-1,1)
            c1=np.append(c1,c)#np.concatenate((c1,c))
        #c2=W@x
        #c=np.array([c1, c2])
        cg=np.append(cg,c1)
        cg=cg.reshape(-1,1)
        #print("CG SHAPE= ", cg.shape)
        return cg#np.concatenate((cg,c1))#np.array([x[1],x[2],x[3],x[4]])))#cg

    def jacobian(self, x): #evaluates jacobian of constraints at x
        #
        # The callback for calculating the Jacobian
        #
        print(np.linalg.matrix_rank(nd.Jacobian(self.constraints)(x)))
        # print(nd.Jacobian(self.constraints)(x).shape)
        return nd.Jacobian(self.constraints)(x)

    # def hessianstructure(self):
    #     #
    #     # The structure of the Hessian
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #
        
    #     return np.eye(6)

    def hessian(self, x, lagrange, obj_factor): #returns hessian of objective function
        #
        # The callback for calculating the Hessian
        # Note:
        #
        #
        Q,q=getQ(self.s, self.mode)
        return self.factor*Q


def runopt(sim: mj.MjSim, s:int, mode:int):         # runs optimization with s number of edges in linearized friction cone and mode which selects PCWF of SFC
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
        # print("rigt finger f :", x[0])
        # print("original direction r :", oldr)
        # print("new direction r :",  fr)
        # print("left finger f :", x[3])
        # print("original direction l :", oldl)
        # print("new direction l :", fl)
        projr=frr@oldr  #project new force onto old one 
        projl=fll@oldl 
        #print("projected on r: ", projr)
        #print("projected on l: ", projl)

    elif mode==1:
        #print("rigt finger f :", x[0:4])
        #print("left finger f :", x[4:8])
        frr=x[0:3]
        fr=frr/np.linalg.norm(frr)
        fll=x[4:7]
        fl=fll/np.linalg.norm(fll)
        oldr,oldl=getFdir(sim)
        # print("rigt finger f :", x[0])
        # print("original direction r :", oldr)
        # print("new direction r :",  fr)
        # print("left finger f :", x[4])
        # print("original direction l :", oldl)
        # print("new direction l :", fl)
        projr=frr@oldr
        projl=fll@oldl
        # print("projected on r: ", projr)
        # print("projected on l: ", projl)
    
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


    
    # index=askGr()
    # sim.set_state(statearr[index])
    # setp(pos[index], quat[index])
    # noth(sim, mj_viewer, 20)


    file1 = open('state-set', 'rb') #open('opt-states', 'rb')
    states = pickle.load(file1)
    file1.close()
    file2 = open('pos-set', 'rb') #open('opt-positions', 'rb')
    poss = pickle.load(file2)
    file2.close()
    file3 = open('quat-set', 'rb') #open('opt-quaternions', 'rb')
    quat = pickle.load(file3)
    file3.close()
    rig=[]
    lef=[]
    failed=[]
    for k in range(100):
        sim.set_state(states[k])
        setp(poss[k], quat[k])
        sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')-0.0017)
        sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')-0.0017)
        noth(sim, mj_viewer, 15)
        #p1,p2=getcontactpos(sim)
        R1,R2=checkside(sim)
        if R1.shape==(3,3) and R2.shape==(3,3):
        
            cr,cl= runopt(sim, 8, 1)
            rig.append(cr)
            lef.append(cl)
        else:
            failed.append(k)


    fileR = open('Rcalc', 'wb') #open('RF-opt', 'wb')
    pickle.dump(rig, fileR) # dump information to that file
    file1.close()# close the file
    fileL = open('Lcalc', 'wb') #open('LF-opt', 'wb')
    pickle.dump(lef, fileL) # dump information to that file
    fileL.close()# close the file
    fileF = open('Fail', 'wb')#open('Fail-opt', 'wb')
    pickle.dump(failed, fileF) # dump information to that file
    fileF.close()# close the file
    # file1 = open('state-set', 'rb')
    # states = pickle.load(file1)
    # file1.close()
    # file2 = open('pos-set', 'rb')
    # positions = pickle.load(file2)
    # file2.close()
    # file3 = open('quat-set', 'rb')
    # quaternions = pickle.load(file3)
    # file3.close()
    # rig=[]
    # lef=[]
    # fcount=0
    # #i=3
    # failed=[]
    # for i in range(100):
    #     sim.set_state(states[i])
    #     sim.data.mocap_pos[0][:]=positions[i] + np.array([0.0 , 0.0, 0.066])
    #     #print("Position: ", positions[i])
    #     sim.data.mocap_quat[0][:]=quaternions[i]
    #     sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')-0.0017)
    #     sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')-0.0017)
    #     noth(sim, mj_viewer, 20)
    #     R1,R2=checkside(sim)
    #     if np.allclose(R1, np.zeros(3)):
    #         fcount+=1
    #         failed.append(i)
    #     else: 
    #         #cr,cl= runopt(sim, 6, 1)
    #         #rig.append(cr)
    #         #cl=3
    #         lef.append(cl)

    # # fileR = open('Rcalc', 'wb')
    # # pickle.dump(rig, fileR) # dump information to that file
    # # file1.close()# close the file

    # # fileL = open('Lcalc', 'wb')
    # # pickle.dump(lef, fileL) # dump information to that file
    # # fileL.close()# close the file
    # fileF = open('Fail', 'wb')
    # pickle.dump(failed, fileF) # dump information to that file
    # fileF.close()# close the file
    # print(failed)
    # print("Nr failed optimizations: ", fcount)

    print("successfully completed")

    # mode, s= askVar()

    # x0=init(sim, s, mode)

    # print(x0)
    # #x0=np.array([1, 0, 0, 1, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0])
    # #x0=np.array([20, 0, 10, 20, 0, 10]).T
    # lb, ub= createBounds(s, mode)
    # #cl=np.array([-1e-7, 0, 0])
    # cl,cu=getConst(mode, s)
    # #print(Quad(s,mode).constraints(x0))
    # solver=ipopt.problem(n=len(x0),
    #             m=len(cl),
    #             problem_obj=Quad(s, mode),
    #             lb=lb,
    #             ub=ub,
    #             cl=cl,
    #             cu=cu
    #             )

    # solver.addOption('tol', 1e-9)
    # #solver.addOption('acceptable_iter', 10)
    # solver.addOption('acceptable_tol', 1e-8)
    # #solver.add_option('mu_strategy','adaptive')
    # solver.addOption('max_iter', 1090)
    # # x,info =solver.solve(x0)
    # x,info =solver.solve(x0)#np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
    # #G=np.array([[1, 0, 0, -1, 0, 0],[0, 1, 0, 0, -1, 0], [0, 0, 1, 0, 0, 1],[0, 0, 0, 0, 0, 0],[0, 0, -1, 0, 0, 1],[0, 1, 0, 0, 1, 0]])
    # #print(x)
    # #print(Quad(s,mode).constraints(x))
    # print("x= ")
    # print(x)
    # print("Constraints evaluated at x: ")
    # print(Quad(s,mode).constraints(x))
    # if mode==0:
    #     frr=x[0:3]
    #     fr=frr/np.linalg.norm(frr)
    #     fll=x[3:6]
    #     fl=fll/np.linalg.norm(fll)
    #     oldr,oldl=getFdir(sim)
    #     print("rigt finger f :", x[0])
    #     print("original direction r :", oldr)
    #     print("new direction r :",  fr)
    #     print("left finger f :", x[3])
    #     print("original direction l :", oldl)
    #     print("new direction l :", fl)
    #     projr=frr@oldr
    #     projl=fll@oldl
    #     print("projected on r: ", projr)
    #     print("projected on l: ", projl)

    # elif mode==1:
    #     #print("rigt finger f :", x[0:4])
    #     #print("left finger f :", x[4:8])
    #     frr=x[0:3]
    #     fr=frr/np.linalg.norm(frr)
    #     fll=x[4:7]
    #     fl=fll/np.linalg.norm(fll)
    #     oldr,oldl=getFdir(sim)
    #     print("rigt finger f :", x[0])
    #     print("original direction r :", oldr)
    #     print("new direction r :",  fr)
    #     print("left finger f :", x[4])
    #     print("original direction l :", oldl)
    #     print("new direction l :", fl)
    #     projr=frr@oldr
    #     projl=fll@oldl
    #     print("projected on r: ", projr)
    #     print("projected on l: ", projl)

    # sim.data.qfrc_applied[6] = -projr#rechts
    # sim.data.qfrc_applied[7] = -projl#links
    # sim.data.mocap_pos[0][:]+= np.array([0, 0, 0.15])
    # noth(sim, mj_viewer, 150)
#T,W=getTW(4,0)
#F=createF(2,4)
#print(T@x-F@W@x)
#print(0.5*x[0]-x[2]-x[1])
#print(G@x.T)
#print(0.5*(x[0]**2 +x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2) +np.sum(x))
#print(info)
#print(help(cyipopt))
