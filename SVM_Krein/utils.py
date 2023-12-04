import numpy as np
import quadprog
#%% 
def quadprog_solve_qp(P,y,C,m=None): # , q, G=None, h=None, A=None, b=None):
    if m is None:
        m = P.shape[0]
    dtype = P.dtype
    q = -np.ones((m, ))
    # print('q_dtype: {}'.format(q.dtype))
    G = np.vstack((np.eye(m)*-1,np.eye(m)))
    h = np.hstack((np.zeros(m), np.ones(m) * C))

    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    A = y.astype(dtype).reshape(1,-1)
    b = 0.0
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    try:
        alpha = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0] 
    except:
        alpha = quadprog.solve_qp(np.eye(m,dtype=np.float64), qp_a, qp_C, qp_b, meq)[0] 
    return alpha

def Krein_EIG(K):
    d,U = np.linalg.eig(K)
    D = np.diag(d)
    S = np.sign(D)
    try:
        K_tilde = (U@S@D@np.linalg.inv(U + 1e-6)).real    
    except:
        K_tilde = (U@S@D@U.T).real
    K_tilde = K_tilde.astype(np.float64)
    return K_tilde,U,S,D