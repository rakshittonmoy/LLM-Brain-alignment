from scipy.stats import spearmanr
import numpy as np

def sanity_check_rdm(name, rdm):
    print(f"\n{name} RDM Summary:")
    print(" - Shape:", rdm.shape)
    print(" - NaNs:", np.isnan(rdm).sum())
    print(" - Infs:", np.isinf(rdm).sum())
    print(" - Unique values:", np.unique(rdm).shape[0])
    print(" - Std deviation:", np.std(rdm))

def compute_rsa(brain_rdm, model_rdm):
    brain_rdm = np.asarray(brain_rdm).flatten()
    model_rdm = np.asarray(model_rdm).flatten()

    valid = np.isfinite(brain_rdm) & np.isfinite(model_rdm)
    if np.std(brain_rdm[valid]) == 0 or np.std(model_rdm[valid]) == 0:
        print("Invalid or constant vectors — returning NaN")
        return np.nan, np.nan

    corr, pval = spearmanr(brain_rdm[valid], model_rdm[valid])
    
    return corr, pval

