# Brain-tuning Improves Generalizability and Efficiency of Brain Alignment in Speech Models [NeurIPS 2025]
[Paper NeurIPS Page](https://neurips.cc/virtual/2025/loc/san-diego/poster/119924) | [arXiv](https://arxiv.org/abs/2510.21520) | [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/119924.png?t=1764802669.4631288)

Implementation for the paper: Brain-tuning Improves Generalizability and Efficiency of Brain Alignment in Speech Models

## Abstract
Pretrained language models are remarkably effective in aligning with human brain responses elicited by natural language stimuli, positioning them as promising model organisms for studying language processing in the brain. However, existing approaches for both estimating and improving this brain alignment are participant-dependent and highly affected by the amount of data available per participant, hindering both generalization to new participants and population-level analyses. In this work, we address these limitations by introducing a scalable, generalizable brain-tuning method, in which we fine-tune pretrained speech language models to jointly predict fMRI responses from multiple participants. We demonstrate that the resulting brain-tuned models exhibit strong individual brain alignment while generalizing across participants. Specifically, our method leads to 1) a 5-fold decrease in the amount of fMRI data needed to predict brain data from new participants, 2) up to a 50% increase in the overall brain alignment, and 3) strong generalization to new unseen datasets. Furthermore, this multi-participant brain-tuning additionally improves downstream performance on semantic tasks, suggesting that training using brain data from multiple participants leads to more generalizable semantic representations. Taken together, these findings demonstrate a bidirectional benefit between neuroscience and AI, helping bridge the gap between the two fields. 

## Datasets and Preprocessing

 We use the Full Moth Radio Hour [Dataset](https://www.nature.com/articles/s41597-023-02437-z). It can be downloaded from [here](https://openneuro.org/datasets/ds003020). No further preprocessing steps for the fMRI responses are needed because the `derivatives/preprocessed_data` in the dataset is already processed as per the original paper recommendation. For the stimuli, the only preprocessing needed is to make sure that the sampling rate is 16000. 

 To easily use the TextGrids and the TRFiles, we recommend downloading them from [Antonello et. al, 2024](https://utexas.app.box.com/v/EncodingModelScalingLaws). 
