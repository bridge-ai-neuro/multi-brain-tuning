import numpy as np  
import config 
import torch 
import matplotlib.pyplot as plt 
from extract_speech_features import extract_speech_features
# from wav2vec_sim import Wav2VecLinear 
from models import Wav2VecLinear, Wav2VecLoRA

from utils.brn_eval_utils import *
from ridge_utils.interpdata import lanczosinterp2D
import ridge_utils.npp
from ridge_utils.ridge import bootstrap_ridge
from ridge_utils.util import make_delayed
import pickle
from glob import glob
from argparse import ArgumentParser


def get_wav_features(model,
                     feature_extractor,
                     model_config,
                     story_name,
                     subject,
                     chunksz=100,
                     contextsz=16000,
                     layers=config.LAYERS_TO_SAVE,
                     batch_size=32,
                     device='cuda'):
    # Load the story data
    wav_tensor, fmri_tensor = load_story_data(story_name, subject)
    # Extract speech features
    chunksz_sec = chunksz / 1000.

    # context size in terms of chunks
    assert (contextsz % chunksz) == 0, "These must be divisible"
    contextsz_sec = contextsz / 1000.
    
    model_config =  {
        "huggingface_hub": config.MODEL_CKPT[args.model_name],
        "stride": 320,
        "min_input_length": 400
    }
    
    extract_features_kwargs = {'model': model, 'feature_extractor': feature_extractor, 'model_config': model_config,
        'wav': wav_tensor, 'sel_layers': layers,
        'chunksz_sec': chunksz_sec, 'contextsz_sec': contextsz_sec,
        'require_full_context': False,
        'batchsz': 1, 'return_numpy': False
    }
    
    
    wav_features = extract_speech_features(**extract_features_kwargs) # keys {'final_outputs': out_features, 'times': times: 'module_features': module_features}
    print(wav_features['final_outputs'].shape)
    # print(wav_features['module_features'])
    return wav_features, fmri_tensor

def predict_stories(subject, model_path=None, layers=config.LAYERS_TO_SAVE, model_pref='base', save_dir=config.PREDS_SAVE_DIR, batch_size=32, device='cuda'):

    if model_path is not None:
        if 'wembed' in model_path.split('/')[-2]: #TODO: don't need it anymore cause out_dim doesnt matter?
            print('wembed model')
            model, processor, model_config, nc_mask = load_model_wembed(model_path, device)
        elif 'lora' in model_path.split('/')[-2]:
            model, processor, model_config, nc_mask = load_lora_model(model_path, subject, device)
        else:
           model, processor, model_config, nc_mask = load_model(model_path, subject, device)
    else:
        model, processor, model_config, nc_mask = load_model(model_path, subject, device)
        
    layers_outputs = {}; final_outputs = {}; fmri_tensors = {}
    for story_name in tqdm(train_stories + test_stories):
        print(f'Processing {story_name}')
        wav_features, fmri_tensor = get_wav_features(model, processor, model_config, story_name, subject, layers=layers, batch_size=batch_size, device=device)
        
        downsampled_features = lanczosinterp2D(wav_features['final_outputs'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times) # downsample features
        final_outputs[story_name] = downsampled_features; fmri_tensors[story_name] = fmri_tensor.numpy()
        
        for layer in layers:
            if layer not in layers_outputs:
                layers_outputs[layer] = {story_name: lanczosinterp2D(wav_features['module_features'][f'layer.{layer}'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times) }
            
            layers_outputs[layer][story_name] =  lanczosinterp2D(wav_features['module_features'][f'layer.{layer}'].cpu().numpy(), wav_features['times'].numpy()[:, 1], wordseqs[story_name].tr_times)

    pred_res_final = predict_encoding(final_outputs, fmri_tensors, subject)
    os.makedirs(f'{save_dir}/UTS0{subject}', exist_ok=True)
    with open(f'{save_dir}/UTS0{subject}/{model_pref}_pred_res_final.pkl', 'wb') as f:
        pickle.dump(pred_res_final, f)
    
    for layer in layers:
        pred_res_layer = predict_encoding(layers_outputs[layer], fmri_tensors, subject)
        with open(f'{save_dir}/UTS0{subject}/{model_pref}_pred_res_layer_{layer}.pkl', 'wb') as f:
            pickle.dump(pred_res_layer, f)
    

def predict_encoding(features, fmri_tensors, subject, trim_start=50, trim_end=5): 
        
    #Training data
    Rstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(features[story][10:-5]) for story in train_stories]))
    #Test data
    Pstim = np.nan_to_num(np.vstack([ridge_utils.npp.zs(features[story][trim_start:-trim_end]) for story in test_stories]))

    # Add FIR delays
    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)
    alphas = np.logspace(1, 4, 15) # Equally log-spaced ridge parameters between 10 and 10000. 
    nboots = 3 # Number of cross-validation ridge regression runs. You can lower this number to increase speed.

    ## Get response data
    Rresp = np.vstack([fmri_tensors[story] for story in train_stories])
    Presp = np.vstack([fmri_tensors[story][40:] for story in test_stories])
    print(f'stim data shapes: {delRstim.shape}, {delPstim.shape}, fmri data shapes: {Rresp.shape}, {Presp.shape}')

    # Bootstrap chunking parameters
    chunklen = 20
    nchunks = int(len(Rresp) * 0.25 / chunklen)

    # Run ridge regression - this might take some time
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, Rresp, delPstim, Presp,
                                                        alphas, nboots, chunklen, nchunks,
                                                        use_corr=False, single_alpha=False)
    
    pred = np.dot(delPstim,  wt)
    dtest = load_fmri_story(test_stories[0], subject)
    
    SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(dtest['individual_repeats'][:, 40:, :], pred, max_flooring=0.25) # NC via computing the SPE and CC
    # res_dict = dict(pred=pred, wt=wt, corr=corr, c_norm=cc_norm, cc_max=cc_max, corrs_unnorm=corrs_unnorm,)
    res_dict = dict(corr=corr, c_norm=cc_norm, cc_max=cc_max, corrs_unnorm=corrs_unnorm,)
    
    
    return res_dict
    
if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--subject', type=int, default=1, help='Subject number')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training')
    parser.add_argument('--logs_dir', type=str, default='wav2vec_multiple_lora_linear_1e4_anat_rois_func_d6_mse_sbind', help='Device to use for training')
    parser.add_argument('--base_test', action='store_true', help='Whether to test the base pretrained model')
    args = parser.parse_args()
    desired_epochs = config.DESIRED_EPOCHS
    print(args)
    nstories = 25; lora_rank = config.LORA_RANK
    train_st_dict = {25:train_stories_25, 20:train_stories_20, 15:train_stories_15, 10:train_stories_10, 5:train_stories_5, 3:train_stories_3, 2:train_stories_2,}
    train_stories = train_st_dict[nstories]
    
    subject = args.subject
    device = args.device
    base_test = args.base_test
    
    model_paths = glob(f'{config.CKPT_SAVE_DIR}/{args.logs_dir}/*')
    model_paths = (sorted(model_paths, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]), reverse=False))
    model_paths = (model_paths)

    if base_test:
        march = f"wav2vec_story_base_subj_{subject}"
        if nstories == 25:
            model_pref = f"mp_{march}"
        else:
            model_pref = f"mp_{nstories}st_{march}"
        predict_stories(subject=subject, model_path=None, model_pref=model_pref, batch_size=1024, device=device, save_dir=config.PREDS_SAVE_DIR)
    
    else:

        for model_path in model_paths:
            epoch = int(model_path.split('/')[-1].split('_')[-1].split('.')[0])
            
            if epoch not in desired_epochs:
                continue
            print(f'Processing {model_path}')
            march = f'{args.logs_dir}_s{subject}' 
            if nstories == 25:
                # march = f'{args.logs_dir}_s{subject}'
                model_pref = f"mp_{march}_epoch_{epoch}_subj_{subject}"
            else:
                model_pref = f"mp_{nstories}st_{march}_epoch_{epoch}_subj_{subject}"

            predict_stories(subject=subject, model_path=model_path, model_pref=model_pref, batch_size=1024, device=device, save_dir=config.PREDS_SAVE_DIR)

