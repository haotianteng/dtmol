"""
    Preprocessing for the SO(3) sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
    Code modified from https://github.com/gcorso/DiffDock/blob/main/utils/so3.py
"""
import os
import numpy as np
import torch
import dtmol
from functools import lru_cache
from scipy.spatial.transform import Rotation

MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 1000
X_N = 2000
omegas = np.linspace(0, np.pi, X_N + 1)[1:]
DATA_FOLDER=dtmol.__path__[0]+"/assets"

def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only (f(w) term in equation 3 in diffDock paper)
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * np.sin(omega * (l + 1 / 2)) / np.sin(omega / 2)
    return p

def _density(expansion, omega, marginal=True):  
    # if marginal, density over [0, pi], else over SO(3) the p(w) term in equation 3 in diffDock paper
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return expansion / 8 / np.pi ** 2  # the constant factor doesn't affect any actual calculations though


def _score(exp, cdf, omega, eps, L=2000):  # score of density over SO(3), the dlog(f(w))/dw term
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2 / 2) * (lo * dhi - hi * dlo) / lo ** 2
    raw_score = dSigma / exp
    #zero out the score to avoid numerical instability
    raw_score[cdf > 1 - 1e-10] = 0
    return raw_score

@lru_cache(maxsize=1)
def _pre_compute():
    """
    This function precalculates the score of the SO(3) distribution under different
    variance (eps) and different angles (omega). The sore function is given in the
    4.3 section in the diffDock paper https://arxiv.org/pdf/2210.01776.pdf  
    """
    global _omegas_array, _cdf_vals, _score_norms, _exp_score_norms, _pdf_vals, _eps_array
    if os.path.exists(os.path.join(DATA_FOLDER,'.so3_omegas_array2.npy')):
        _omegas_array = np.load(os.path.join(DATA_FOLDER,'.so3_omegas_array2.npy'))
        _pdf_vals = np.load(os.path.join(DATA_FOLDER,'.so3_pdf_vals2.npy'))
        _cdf_vals = np.load(os.path.join(DATA_FOLDER,'.so3_cdf_vals2.npy'))
        _score_norms = np.load(os.path.join(DATA_FOLDER,'.so3_score_norms2.npy'))
        _exp_score_norms = np.load(os.path.join(DATA_FOLDER,'.so3_exp_score_norms2.npy'))
    else:
        print("Precomputing and saving to cache SO(3) distribution table")
        _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
        _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

        _exp_vals = np.asarray([_expansion(_omegas_array, eps) for eps in _eps_array]) #f(w) term in equation 3 in diffDock paper
        _pdf_vals = np.asarray([_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals]) #p(w) term in equation 3 in diffDock paper
        _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals]) #normalized cdf of the pdf (pdf sum to pi)
        _score_norms = np.asarray([_score(_exp_vals[i], _cdf_vals[i],_omegas_array, _eps_array[i]) for i in range(len(_eps_array))]) #dlog(f(w))/dw term in equation 3 in diffDock paper

        _exp_score_norms = np.sqrt(np.sum(_score_norms**2 * _pdf_vals, axis=1) / np.sum(_pdf_vals, axis=1) / np.pi)
        #Calculate the expectation of score norm summing over omega
        #sqrt{E[S^2]/pi} S = dlog(f(w))/dw
        np.save(os.path.join(DATA_FOLDER,'.so3_pdf_vals2.npy'), _pdf_vals)
        np.save(os.path.join(DATA_FOLDER,'.so3_omegas_array2.npy'), _omegas_array)
        np.save(os.path.join(DATA_FOLDER,'.so3_cdf_vals2.npy'), _cdf_vals)
        np.save(os.path.join(DATA_FOLDER,'.so3_score_norms2.npy'), _score_norms)
        np.save(os.path.join(DATA_FOLDER,'.so3_exp_score_norms2.npy'), _exp_score_norms)

_pre_compute()

def get_eps_idx(eps):
    """Get the index of the variance in the precomputed table.
    """
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    return eps_idx

def sample(eps):
    """Sample the rotation angle omega given the variance eps.
    """
    eps_idx = get_eps_idx(eps)
    x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)

def sample_vec(eps):
    """Sample the eular vector given the variance eps. The norm of the
    vector is the omega, and the normalized vector is the axis of rotation.
    """
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps, vec):
    """Calculate the score of the eular vector by interpolating from a precomputed
    table.
    Notice here the vector's norm is the omega, and normalized
    vector is the axis of rotation. COuld be 
    Args:
        eps: variance
        vec: eular vector, the norm is the omega, and the normalized vector is the axis of rotation.
    Output:
        A 3-dim vector, the score of the eular vector for each dimension, need to be normalized by
            score_norm when calculating the loss, e.g. rot_loss = (((rot_pred - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims).
    """
    eps_idx = get_eps_idx(eps)
    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om

def get_density(eps, omega):
    """Calulcate the marginal probability P_eps(w) term in the diffDock paper by interpolating from a precomputed table.
    """
    eps_idx = get_eps_idx(eps)
    return np.interp(omega, _omegas_array, _pdf_vals[eps_idx])/X_N * np.pi

def score_norm(eps):
    try:
        eps = eps.numpy()
    except:
        pass
    eps_idx = get_eps_idx(eps)
    return _exp_score_norms[eps_idx]

if __name__ == "__main__":
    """ Test the SO(3) sampling and score computations """
    # omega = sample(2)
    # axis = sample_vec(2)
    # print(score_vec(2, axis))
    # rot_matrix = Rotation.from_rotvec(axis).as_matrix()
    # print(rot_matrix)
    # orig_vec = np.array([1, 0, 0])
    # rot_vec = rot_matrix @ orig_vec

    # # plot the original and rotated vector
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.quiver(0, 0, 0, orig_vec[0], orig_vec[1], orig_vec[2], color='r')
    # ax.quiver(0, 0, 0, rot_vec[0], rot_vec[1], rot_vec[2], color='b')
    # #plot the eular vector
    # ax.quiver(0, 0, 0, axis[0], axis[1], axis[2], color='g')

    # 3D plot the sampled unit vector
    from matplotlib import pyplot as plt
    from angle_plot import plot_angle_distribution_polar
    for i in np.arange(100,1000,100):
        plt.plot(_omegas_array,_score_norms[i], color = [i/1000,0,i/1000], label = f"eps = {10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)[i]:.2f}")
    plt.legend()
    plt.xlabel("omega")
    plt.ylabel("score")
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)

    sampled = []
    for _ in range(10000):
        axis = sample_vec(2)
        axis = axis / np.linalg.norm(axis)
        sampled.append(axis)
    # 3d plot the end points of the sampled unit vector
    sampled = np.array(sampled)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2])
    plot_angle_distribution_polar(sampled, np.array([0, 0, 1]))
    plt.show()

    #plot the distribution of scores
    scores_eps = []
    es = []
    vecs = []
    corrs = []
    for eps_idx in range(1,1000):
        eps = _eps_array[eps_idx]
        es.append(eps)
        vec = sample_vec(eps)
        vecs.append(vec)
        score = score_vec(eps, vec)/score_norm(eps)
        corr = np.dot(vec, score)/np.linalg.norm(vec)/np.linalg.norm(score)
        corrs.append(corr)
        scores_eps.append(np.linalg.norm(score))
    fig = plt.figure()
    plt.plot(es, scores_eps)
    plt.xlabel("epislon")
    plt.ylabel("score")
    
    #plot the score norm 
    score_norms = []
    es = []
    for eps in np.arange(0,1,0.01):
        score_norms.append(score_norm(eps))
        es.append(eps)
    fig = plt.figure()
    plt.plot(es, score_norms)

    #plot the normalized score
    scores_eps = []
    es = []
    for eps in range(1,100):
        es.append(eps)
        score = score_vec(eps, sample_vec(eps))/score_norm(eps)
        scores_eps.append(np.linalg.norm(score))
    fig = plt.figure()
    plt.plot(es, scores_eps)
    plt.show()