"""Methods and metrics reported in the completed experimental study."""
DATASETS = ['3cad','mvtec_ad2']
METRICS = ['i_auroc','p_auroc','p_ap','aupro']
PRIMARY = 'denoising_vmf_p25'
METHOD_NAME = 'SDD-RD'
SEEDS = [17,29,43]
METHODS = [('rd','RD'),('rdpp','RD++'),('patchcore','PatchCore'),('dinomaly','Dinomaly (matched)'),(PRIMARY,METHOD_NAME)]
ABLATIONS = [('dinomaly','Cosine'),('vmf_fixed','Fixed-concentration vMF'),('vmf_single','Single vMF'),
             ('denoising_cosine_p25','Cosine + denoising'),(PRIMARY,'Single vMF + denoising'),
             ('vmf_mixture','Uniform vMF mixture'),('context_mixture','Contextual vMF mixture'),
             ('coupled_mixture','Coupled vMF mixture')]


def seeds_for(variant):
    return SEEDS if variant in ['dinomaly',PRIMARY,'denoising_cosine_p25','coupled_mixture'] else [17]
