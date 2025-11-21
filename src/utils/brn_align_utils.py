import cortex
import config
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import mne
from nilearn import datasets, surface
import pickle
import statsmodels
import nibabel as nib
subjects_dir = mne.datasets.sample.data_path() / 'subjects'

labels_f6 = mne.read_labels_from_annot(
    'fsaverage6', 'HCPMMP1', 'both', subjects_dir=subjects_dir)

fs_version = 'fsaverage'
datasets.fetch_surf_fsaverage(fs_version)

subjects_dir = mne.datasets.sample.data_path() / 'subjects'
labels = mne.read_labels_from_annot(
    fs_version, 'HCPMMP1', 'both', subjects_dir=subjects_dir)

# load Noise Ceilings
NCs = {}
for x in os.listdir('../datasets/subject_NCs'):
    subj_data = np.load(('../datasets/subject_NCs/'+x))
    subj = int(x.split('_')[0].split('.')[0].split('UTS')[1])
    NCs[subj] = subj_data

indices_dict = pickle.load(open('../datasets/subject_NCs/late_language_rois.p', 'rb'))

early_visual_rois = ['V1','V2']
early_indices = [indices_dict[r] for r in early_visual_rois]

vwfa_rois = ['PH','TE2P']
vwfa_indices = [ 111, 112]


early_auditory_rois = ['A1','LBelt','PBelt','MBelt','RI','A4']
early_auditory_indices = [indices_dict[r] for r in early_auditory_rois]

primary_auditory_rois = ['A1']
primary_auditory_indices = [indices_dict[r] for r in primary_auditory_rois]

secondary_auditory_rois = ['STV']
secondary_auditory_indices = [indices_dict[r] for r in secondary_auditory_rois]

late_language_rois = ['A5','44','45', 'IFJa', 'IFSp', 'PGi', 'PGp', 'PGs', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'STGa','STSda', 'STSdp','TA2', 'STSva', ]
late_language_indices = [indices_dict[r] for r in late_language_rois]

main_language_rois = ['EV','VWFA','EAC','LL','A1']
angular_gyrus_rois = ['PGi', 'PGp', 'PGs', 'TPOJ2', 'TPOJ3']
ltc_rois = ['TPOJ1', 'STGa','STSda', 'STSdp','A5', 'STSva', ]
ifg_mfg_rois = ['44','45', 'IFJa', 'IFSp']

angular_gyrus_indices = [indices_dict[r] for r in angular_gyrus_rois]
ltc_indices = [indices_dict[r] for r in ltc_rois]
ifg_mfg_indices = [indices_dict[r] for r in ifg_mfg_rois]

main_language_indices = [early_indices, vwfa_indices, early_auditory_indices, late_language_indices, primary_auditory_indices, angular_gyrus_indices, ltc_indices, ifg_mfg_indices]




from nilearn import datasets, surface
fsaverage = datasets.fetch_surf_fsaverage(fs_version)
def get_hm_surf(im):
    surf_data_lh = surface.vol_to_surf(
        im,
        surf_mesh=fsaverage["pial_left"],
        inner_mesh=fsaverage["white_left"],
        interpolation="linear" 
    )

    surf_data_rh = surface.vol_to_surf(
        im,
        surf_mesh=fsaverage["pial_right"],
        inner_mesh=fsaverage["white_right"],
        interpolation="linear" )
    
    return surf_data_lh, surf_data_rh

def get_nib_im(vol,
               subj,
               pdir='../datasets/affine_transforms'):
    affine = np.load(f'{pdir}/affine_s{subj}.npy')
    nib_im = nib.Nifti1Image(vol.volume.data.squeeze().T, affine)
    return nib_im


def get_norm_lan_scores(surf_data_lh, surf_data_rh, model_name='wav2vec'):
    normalized_scores_text_lan = {model_name:[]}

    for eachroi in np.arange(len(main_language_indices)):
        temp2 = []
        for subroi in main_language_indices[eachroi]:
            temp1 = []
            temp = []
            # print(model_name, labels[subroi])
            lh = np.nan_to_num(surf_data_lh[labels[subroi].vertices])
            rh = np.nan_to_num(surf_data_rh[labels[subroi + 180].vertices ])

            lh_rh = np.concatenate([lh,rh],axis=0)
            temp.append(np.mean(lh_rh))

            temp1.append(np.mean(temp))
            temp2.append(np.array(temp1))
        normalized_scores_text_lan[model_name].append(np.mean(np.array(temp2),axis=0))
    normalized_scores_text_lan = np.array(list(normalized_scores_text_lan[model_name]))
    return normalized_scores_text_lan.squeeze()




def get_subjs_layers_base(subj_list=[1, 2, 3], layers=[2, 7, 8, 10, 12], model_name='wav2vec',):
    region_keys = ['early_visual', 'VWFA', 'early_auditory', 'late_language', 'primary_auditory']
    base_scores = {k:{s:{} for s in subj_list}  for k in region_keys }    
    # print(subj_scores, base_scores)
    for subj in subj_list:
        for layer in layers:
            norm_scores_base = get_norm_scores_base(subj, layer, model_name=model_name)
            for i, k in enumerate(region_keys):
                base_scores[k][subj][layer] = (norm_scores_base[i])  
        
    return  base_scores

def add_region_keys(score):
    region_keys = ['early_visual', 'VWFA', 'early_auditory', 'late_language', 'primary_auditory']
    region_keys_dict = {}
    for i, k in enumerate(region_keys):
        region_keys_dict[k] = score[i]
    return region_keys_dict


def get_fil_acc(corrs_unnorm, cc_max, cc_thr=0.4, roi_inds=None):
    acc = []
    if roi_inds:
        corrs_unnorm = corrs_unnorm[roi_inds]; cc_max = cc_max[roi_inds]
        
    for ei, i in enumerate(cc_max):
        if i <= cc_thr:
            acc.append(np.nan)
        else:
            acc.append(corrs_unnorm[ei])
    acc = np.array(acc)
    return acc

def get_vol_base_(subj, layer, prefix, read_dir=config.PREDS_SAVE_DIR, cc_thr=0.05):
    ly = layer
    base_prds = pickle.load(open(f'./{read_dir}/UTS0{subj}/{prefix}_s{subj}_{subj}_pred_res_layer_{ly}.pkl', 'rb'))
    base_arr = get_fil_acc(base_prds['corrs_unnorm'], base_prds['cc_max'], cc_thr) 
    
    diff_arr = base_arr
    
    vol = cortex.Volume(np.nan_to_num(diff_arr/NCs[subj]), f'UTS0{subj}', f'UTS0{subj}_auto')
    nib_im = get_nib_im(vol, subj)
    
    return nib_im

def get_vol_hm_(subj, layer, prefix):
    ly = layer
    im = get_vol_base_(subj, layer, prefix)
    surf_data_lh, surf_data_rh = get_hm_surf(im)
    return surf_data_lh, surf_data_rh


def get_norm_scores_base(subj, layer, prefix,):
    surf_data_lh_b, surf_data_rh_b = get_vol_hm_(subj, layer, model_name=prefix)
    norm_scores_base = get_norm_lan_scores(surf_data_lh_b, surf_data_rh_b, model_name=config.PRED_MODEL_NAME)
    
    return norm_scores_base


def get_subjs_layers_base(subj_list=[1, 2, 3, 6], prefix='mean_mp',):
    region_keys = ['early_visual', 'VWFA', 'early_auditory', 'late_language', 'primary_auditory', 'angular_gyrus', 'ltc', 'ifg_mfg']
    base_scores = {k:{s:{} for s in subj_list}  for k in region_keys }    
    for subj in subj_list:
        for layer in config.LAYERS_TO_SAVE:
            norm_scores_base = get_norm_scores_base(subj, layer, prefix=prefix, )
            for i, k in enumerate(region_keys):
                base_scores[k][subj][layer] = (norm_scores_base[i])  
        
    return  base_scores

