import numpy as np
import ot

def ICP_tracts(m1,m2,reflexion=True,lam1=0,lam2=0,max_itr=5,eps=1e-7): 
    d=m1.shape[-1]
    nb_streamline=m1.shape[0]
    P=np.eye(nb_streamline)
    R=np.eye(d)
    
    if reflexion:
        m1,m2=reflexion_Ox(m1,m2)
        
    m1_mean,m2_mean=np.mean(m1,axis=(0,1)),np.mean(m2,axis=(0,1))
    m1,m2=m1-m1_mean,m2-m2_mean
    
    for i in range(max_itr-1):
        print(i,end=' ',flush=True)
        m1,m2,delta_P=assignement_step(m1,m2)
        m1,m2,delta_R=procruste_step(m1,m2,lam1)
        R=R@delta_R
        P=delta_P@P
        
        #Check convergence
        diff=np.linalg.norm(delta_R-np.eye(d))
        if diff <eps:
            break

    m1,m2=m1+m1_mean,m2+m2_mean
    m1,m2,t=shift_step(m1,m2,lam2)    
    return m1,m2,R,t,np.nonzero(P)[1]

def reflexion_Ox(m1,m2):
    d=m1.shape[-1]
    m1_reshape=m1.reshape((-1,d))
    R=np.eye(d)
    R[0,0]=-1
    return (m1_reshape@R).reshape(m1.shape),m2
    
def assignement_step(m1,m2):
    nb_streamline=m1.shape[0]
    C=np.sum((m1[:,None]-m2[None])**2,(2,3))
    w=np.ones((nb_streamline,))
    P=ot.emd(w,w,C)
    return m1,m2[np.nonzero(P)[1]],P
    
def procruste_step(m1,m2,lam1):
    d=m1.shape[-1]
    m1_reshape,m2_reshape=m1.reshape((-1,d)),m2.reshape((-1,d))
    R=procruste_superimposition(m1_reshape,m2_reshape,lam1=lam1)
    return (m1_reshape@R).reshape(m1.shape),m2,R
    
def shift_step(m1,m2,lam2=0):
    t=np.mean(m1-m2,axis=(0,1))/(lam2+1)
    return m1-t,m2,t

def procruste(m1, m2,lam1=0):
    d = m1.shape[-1]
    
    A = m2.T@m1 # size (d,d)
    U, S, V = np.linalg.svd(A+lam1*np.eye(d), full_matrices=True)
    R = (V.T) @ (U.T)
    return R
    
def procruste_superimposition(m1, m2,lam1=0): #Impose R to be a rotation (<=>detR=1)
    d = m1.shape[-1]
    
    A = m2.T@m1 # size (d,d)
    U, S, V = np.linalg.svd(A+lam1*np.eye(d), full_matrices=True)
    R = (V.T) @ (U.T)
    
    if np.linalg.det(R) < 0:
        I=np.eye(d)
        I[-1,-1]=-1
        R = (V.T)@I@ (U.T)
    return R
    
    
