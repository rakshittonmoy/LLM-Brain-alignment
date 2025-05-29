import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_network_activations(fmri_mat, network_name):
    networks = [atlas[0] for atlas in fmri_mat['meta']['atlases'][0][0][0]]
    idx = networks.index(network_name)
    nw_columns = fmri_mat['meta']['roiColumns'][0][0][0][idx]
    column_indexes = np.concatenate([nw_columns[roi][0].flatten() - 1 for roi in range(len(nw_columns))])
    return fmri_mat['examples'][:, column_indexes]

def build_rdm(data):
    rdm = squareform(pdist(data, metric='cosine'))
    rdm_up_tri = rdm[np.triu_indices(rdm.shape[0], k=1)]
    return rdm, rdm_up_tri
