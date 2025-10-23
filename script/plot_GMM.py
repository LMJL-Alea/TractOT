# Draw GMM with/without streamlines 
import numpy as np
import matplotlib.pylab as pl
from matplotlib.patches import Ellipse

def draw_iso_MAP(m, wI,I,color=None, alpha=1,s=1,lim=10,s_iso=None,axis=(0,1),color_scatter=None):

    if color_scatter is None:
        color_scatter=color
    if s_iso is None:
        s_iso=1
        
    iso = pl.Circle((m[axis[0]], m[axis[1]]),s_iso*I,color=color,alpha=alpha * wI)
    pl.gca().add_artist(iso)
    
    pl.scatter(m[axis[0]], m[axis[1]], color=color_scatter,s=s,alpha=alpha)    
    pl.xlim(-lim,lim)
    pl.ylim(-lim,lim) 
    
def draw_aniso_MAP(m,w,S, color=None, alpha=1,s=1,lim=10,s_aniso=None,axis=(0,1),color_scatter=None):

    if color_scatter is None:
        color_scatter=color    
    if s_aniso is None:
        s_aniso = w.shape[0]*[1]
        
    for k in range(w.shape[0]):
        draw_aniso(m,S[k,np.array(axis)[:,None],np.array(axis)],alpha * w[k], color, None,s=s,s_aniso=s_aniso[k],axis=axis)
    
    pl.scatter(m[axis[0]], m[axis[1]], color=color_scatter,s=s,alpha=alpha)    
    pl.xlim(-lim,lim)
    pl.ylim(-lim,lim) 



def eigsorted(Cov):
    vals, vecs = np.linalg.eigh(Cov)
    order = vals.argsort()[::-1].copy()
    return np.maximum(vals[order],0), vecs[:, order]
    
def draw_cov(mu, C, color=None, label=None, nstd=1, alpha=0.5,s=1e2,lim=10):
    vals, vecs = eigsorted(C)
    x_axis=np.zeros(vals.shape)
    x_axis[0]=1
    theta = np.degrees(np.arccos(np.clip(np.dot(vecs[:,0],x_axis), -1.0, 1.0)))
    #theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals[:2])
    pl.scatter(mu[0], mu[1], color=color,s=s,alpha=alpha)
    ell = Ellipse(xy=(mu[0], mu[1]),width=w,height=h,alpha=alpha,angle=theta,facecolor=color,edgecolor=color,label=label,fill=True)
    pl.gca().add_artist(ell)
    pl.xlim(-lim,lim)
    pl.ylim(-lim,lim)


def draw_gmm(m, Cov, w, color=None, nstd=0.5, alpha=1,s=1e2,lim=10):
    for k in range(m.shape[0]):
        draw_cov(m[k], Cov[k], color, None, nstd, alpha * w[k],s=s,lim=lim)
        
def draw_aniso(m,S,alpha=0.5, color=None, label=None ,s=1,s_aniso=None,axis=(0,1)):
    vals, vecs = eigsorted(S)
    x_axis=np.zeros(vals.shape)
    x_axis[0]=1
    theta = np.degrees(np.arccos(np.clip(np.dot(vecs[:,0],x_axis), -1.0, 1.0)))
    w, h = 2 * s_aniso * np.sqrt(vals[:2]) #proj
    #theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    aniso = Ellipse(xy=(m[axis[0]], m[axis[1]]),width=w,height=h,alpha=alpha,angle=theta,facecolor=color,edgecolor=None,label=label,fill=True)
    pl.gca().add_artist(aniso)


   

