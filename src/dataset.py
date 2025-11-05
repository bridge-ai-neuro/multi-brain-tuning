import joblib
from ridge_utils.dsutils import make_word_ds
import torch
from torch.utils.data import Dataset
import numpy as np
ndelays = 4 # We use 4 FIR delays (2 seconds, 4 seconds, 6 seconds, 8 seconds)
delays = range(0, ndelays + 1)
import os
import h5py
grids = joblib.load("../datasets/story_data/grids_huge.jbl") # Load TextGrids containing story annotations
trfiles = joblib.load("../datasets/story_data/trfiles_huge.jbl") # Load TRFiles containing TR information
wordseqs = make_word_ds(grids, trfiles)




def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(reversed(delays)):
        dstim = torch.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.clone()
        dstims.append(dstim)
    return torch.hstack(dstims)

TARGET_SAMPLE_RATE = 16000

def extract_speech_times(   wav: torch.Tensor,
                            chunksz_sec: float=1, contextsz_sec: float=1,
                            num_sel_frames = 1,
                            sampling_rate: int = 16000, require_full_context: bool = False,
                            stereo: bool = False):
    assert (num_sel_frames == 1), f"'num_sel_frames` must be 1 to ensure causal feature extraction, but got {num_sel_frames}. "\
        "This option will be deprecated in the future."
    if stereo:
        raise NotImplementedError("stereo not implemented")
    else:
        assert wav.ndim == 1, f"input `wav` must be 1-D but got {wav.ndim}"

    # Compute chunks & context sizes in terms of samples & context
    chunksz_samples = int(chunksz_sec * sampling_rate)
    contextsz_samples = int(contextsz_sec * sampling_rate)

    # `snippet_ends` has the last (exclusive) sample for each snippet
    snippet_ends = []
    if not require_full_context:
        # Add all snippets that are _less_ than the total input size
        # (context+chunk)
        snippet_ends.append(torch.arange(chunksz_samples, contextsz_samples+chunksz_samples, chunksz_samples))

    # Add all snippets that are exactly the length of the requested input
    # (`Tensor.unfold` is basically a sliding window).
    if wav.shape[0] >= chunksz_samples+contextsz_samples:
        # `unfold` fails if `wav.shape[0]` is less than the window size.
        snippet_ends.append(
            torch.arange(wav.shape[0]).unfold(0, chunksz_samples+contextsz_samples, chunksz_samples)[:,-1]+1
        )

    snippet_ends = torch.cat(snippet_ends, dim=0) # shape: (num_snippets,)

    if snippet_ends.shape[0] == 0:
        raise ValueError(f"No snippets possible! Stimulus is probably too short ({wav.shape[0]} samples). Consider reducing context size or setting `require_full_context=True`")

    # 2-D array where `[i,0]` and `[i,1]` are the start and end, respectively,
    # of snippet `i` in samples. Shape: (num_snippets, 2)
    snippet_times = torch.stack([torch.maximum(torch.zeros_like(snippet_ends),
                                               snippet_ends-(contextsz_samples+chunksz_samples)),
                                 snippet_ends], dim=1)
    
    times_in_seconds = snippet_times / sampling_rate
    return dict(
        indices=snippet_times,
        times=times_in_seconds
    )
    
def get_inds2tr_map_shifted(wav,
                    indices,
                    times,
                    tr_times,
                    used_chuncksz_sec = 1,
                    trim_start = 10,
                    trim_end = 5,
                    expectedtr=2.0045,):
    
    num_snippets = int(expectedtr//used_chuncksz_sec)
    aligned_wav = []
    fmri_inds = []
    # for tx, trtime in enumerate(tr_times[:trim_start]):
    #     if trtime > 0:
    #         start, end = 
    #         aligned_wav.append(wav[:i])
        
    for tridx, trtime in enumerate(tr_times[:]):
        # if trtime < 2:
        #     aligned_wav.append(torch.zeros(32000))
        #     continue
        if trtime < 2:
            aligned_wav.append(torch.cat([torch.zeros(16000), wav[:16000]]))
            continue
        else:
            sidx, eidx = 16000 * int(tr_times[tridx-1]), 16000 * int(tr_times[tridx])
            if wav[sidx:eidx].shape[0] < 32000:
                aligned_wav.append(torch.cat([torch.zeros(32000 - wav[sidx:eidx].shape[0]), wav[sidx:eidx]]))
            else:
                aligned_wav.append(wav[sidx:eidx])
        #     continue

    return torch.vstack(aligned_wav)

class FMRIStory(Dataset):
    def __init__(self,
                 story_name,
                 subject,
                 sub_nc_mask=None,
                 read_stim_dir='../datasets/processed_stim_data_dp',
                 read_fmri_dir='../../ds003020/derivative/transformed_data_fs_lang/', 
                 delays=range(0, ndelays + 1),
                 trim_start=10,
                 trim_end=5,
                 **wav_params):
        super(FMRIStory, self).__init__()
        self.story_name = story_name
        # steps
        ## load wav tensor 
        ## using wav params create dict(inds) -> TRs and dict(TRs) -> inds [that's the main effort LOL]
            ## thinking of 2 ways:
                # - create the normal 16s window and from the downsampple, get the indices of the TRs instead of applying sinc, get unique input window for that
                # - use smaller window, bigger stride, then get the TRs, no averaging
            ## for delays, get dict[TRs] -> inds and concat the wavs 

        ## create a wav tensor of shape (time, feat_size)
        ## create a TR tensor of shape (time, TRs)
        ## load fmri data for the story
        ## create a fmri tensor of shape (time, voxels) to match the wav tensor
        # 
        print(f"Loading {story_name} for subject {subject}")
        self.story_name = story_name
        self.subject = subject
        self.sub_nc_mask = sub_nc_mask
        self.read_stim_dir = read_stim_dir
        self.read_fmri_dir = read_fmri_dir
        self.delays = delays
        self.wav_params = wav_params
        self.wav_tensor = torch.tensor(np.load(os.path.join(self.read_stim_dir, f"{story_name}/wav.npy")))
        self.wav_feat = extract_speech_times(self.wav_tensor, self.wav_params['chunksz_sec'], self.wav_params['contextsz_sec'],
                                             sampling_rate=self.wav_params['sampling_rate'])
        
        self.trim_start = trim_start; self.trim_end = trim_end
        
        # aligned_wav = get_inds2tr_map(self.wav_tensor, self.wav_feat['indices'], self.wav_feat['times'],
        #                               wordseqs[story_name].tr_times, used_chuncksz_sec=self.wav_params['chunksz_sec'])
        
        # self.delayed_wav = make_delayed(aligned_wav, delays)
        # self.fmri_tensor = torch.tensor(self._load_h5py(os.path.join(self.read_fmri_dir,
                                                                    #  f'UTS0{self.subject}', f"{story_name}.hf5")))
        
        # assert self.delayed_wav.shape[0] == self.fmri_tensor.shape[0], "Wav and FMRI tensor should have the same time dimension"
        
    
    def fetch_data(self):
        # print(f'processing {self.story_name} for subject {self.subject}')
        self.aligned_wav = get_inds2tr_map_shifted(self.wav_tensor, self.wav_feat['indices'], self.wav_feat['times'],
                                      wordseqs[self.story_name].tr_times + 1, used_chuncksz_sec=self.wav_params['chunksz_sec'])
        
        self.delayed_wav = make_delayed(self.aligned_wav, self.delays).float()[self.trim_start:-self.trim_end]
        self.fmri_tensor = torch.tensor(self._load_h5py(os.path.join(self.read_fmri_dir,
                                                                     f'UTS0{self.subject}_nonc', f"{self.story_name}.hf5"))).float()
        
        assert self.delayed_wav.shape[0] == self.fmri_tensor.shape[0], "Wav and FMRI tensor should have the same time dimension"
        # print(f'wav tensor shape: {self.delayed_wav.shape}, fmri tensor shape: {self.fmri_tensor.shape}')
    def __getitem__(self, index):
        ## return wav, TR
        ## select from the wav tensor and TR tensor
        return self.delayed_wav[index], self.fmri_tensor[index]
  
    def __len__(self):
        return len(self.delayed_wav)#self.fmri_tensor.shape[0]
    
    def _load_h5py(self, file_path, key=None):   

        data = dict()
        with h5py.File(file_path) as hf:
            if key is None:
                for k in hf.keys():
                    # print("{} will be loaded".format(k))
                    data[k] = list(hf[k])
            else:
                data[key] = hf[key]
        if self.sub_nc_mask:
            return np.nan_to_num(np.array(data['data']))[:, self.sub_nc_mask] # mask out the nans and choose the most sig voxels
        else:
            return  (np.nan_to_num(np.array(data['data'])))

    def _clear(self):
        self.aligned_wav = None
        self.delayed_wav = None
        self.fmri_tensor = None

