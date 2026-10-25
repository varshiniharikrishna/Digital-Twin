"""Federated averaging demo for linear regression; simulated hospitals only."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def _fit_gd(x,y,epochs=100,lr=.03):
    x=np.c_[np.ones(len(x)),x];w=np.zeros(x.shape[1])
    for _ in range(epochs):w-=lr*(2/len(y))*x.T@(x@w-y)
    return w

def run_federated_demo(X,y,k=3,seed=42,rounds=10):
    rng=np.random.default_rng(seed);x=np.asarray(X,float);y=np.asarray(y,float);perm=rng.permutation(len(y))
    split=max(1,int(.8*len(y)));tr=perm[:split];te=perm[split:];parts=np.array_split(tr,k)
    if not len(te):raise ValueError("Federated demo requires at least two rows.")
    mu=x[tr].mean(axis=0);sd=np.where(x[tr].std(axis=0)==0,1,x[tr].std(axis=0));x=(x-mu)/sd
    central=_fit_gd(x[tr],y[tr]);local=_fit_gd(x[parts[0]],y[parts[0]])
    w=np.zeros(x.shape[1]+1)
    for _ in range(rounds):
        ws=[]
        for p in parts:
            p=np.intersect1d(p,tr)
            if len(p):ws.append((len(p),_fit_gd(x[p],y[p],epochs=20)))
        if not ws:break
        w=sum(n*v for n,v in ws)/sum(n for n,_ in ws)
    xt=np.c_[np.ones(len(te)),x[te]]
    return {"origin":"SIMULATED hospital partition; not real federated infrastructure","algorithm":"FedAvg for linear model; trees are not averaged by this demo","n_clients":k,"rounds":rounds,
      "mse":{"centralized":float(mean_squared_error(y[te],np.c_[np.ones(len(te)),x[te]]@central)),"federated":float(mean_squared_error(y[te],xt@w)),"local_only":float(mean_squared_error(y[te],xt@local))},
      "limitations":["random partition is not site heterogeneity","no secure aggregation or differential privacy","comparison uses the final 20% by row after shuffled client assignment"]}
