from scipy.stats import spearmanr
import numpy as np

def compute_rsa(brain_rdm, model_rdm):
    brain_rdm = np.asarray(brain_rdm).flatten()
    model_rdm = np.asarray(model_rdm).flatten()

    if brain_rdm.shape != model_rdm.shape:
        raise ValueError(f"Shape mismatch: brain {brain_rdm.shape}, model {model_rdm.shape}")

    valid_mask = np.isfinite(brain_rdm) & np.isfinite(model_rdm)

    brain_rdm = brain_rdm[valid_mask]
    model_rdm = model_rdm[valid_mask]

    if len(brain_rdm) == 0 or np.std(brain_rdm) == 0 or np.std(model_rdm) == 0:
        print("Invalid or constant vectors — returning NaN")
        return np.nan
    
    print("NaNs in brain:", np.isnan(brain_rdm).sum())
    print("NaNs in model:", np.isnan(model_rdm).sum())

    print("Infs in brain:", np.isinf(brain_rdm).sum())
    print("Infs in model:", np.isinf(model_rdm).sum())
    
    print("Brain std:", np.std(brain_rdm))
    print("Model std:", np.std(model_rdm))

    print("Brain RDM (first 10):", brain_rdm[:10])
    print("Model RDM (first 10):", model_rdm[:10])

    return spearmanr(brain_rdm, model_rdm).correlation

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
        return np.nan
    
    return spearmanr(brain_rdm[valid], model_rdm[valid]).correlation

