import os
import torch
import torch.optim as optim
from argparse import ArgumentParser
import config
from data_utils import *
from train_utils import *
from dataset import FMRIStory
from tqdm import tqdm
import numpy as np 
import time
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import  get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

ndelays = 4 # 4 FIR delays (2 seconds, 4 seconds, 6 seconds, 8 seconds)
delays = range(0, ndelays + 1)

np.random.seed(122)
torch.manual_seed(122)
parser = ArgumentParser()
## model args
parser.add_argument('--model_name', type=str, default='wav2vec', help='Name of the pre-trained model')
parser.add_argument('--loss_name', type=str, default='l2', help='Name of the loss function')

## training args
parser.add_argument('--lora_rank', type=int, default=config.LORA_RANK, help='LoRA rank')
parser.add_argument('--base_lr', type=float, default=config.BASE_LR, help='Learning rate for the base model')
parser.add_argument('--linear_layer_lr', type=float, default=config.LINEAR_LAYER_LR, help='Learning rate for the linear layer')
parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size for training')
parser.add_argument('--device', type=str, default=config.DEVICE, help='Device to use for training')

## data args
parser.add_argument('--subject', type=int, default=config.SUBJECT, help='Subject number to train on')
parser.add_argument('--sampling_rate', type=int, default=config.SAMPLING_RATE, help='Sampling rate of the audio')

## logs args
parser.add_argument('--save_dir', type=str, default=config.SAVE_DIR)
parser.add_argument('--save_name', type=str, default=config.SAVE_NAME)
parser.add_argument('--exp_name', type=str, default=config.EXP_NAME) 

args = parser.parse_args()
print(args)
        
if __name__ == '__main__':
    print(f"using model name {args.model_name}")
    os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'logs', args.exp_name), exist_ok=True)
    train_stories = list(np.load('../datasets/story_lists.npy')) #list(wordseqs.keys())
    train_stories.remove(test_stories[0]); train_stories.remove(val_stories[0])
            
    device = args.device; num_epochs = args.num_epochs
    model = get_model(model_ckpt=config.MODEL_CKPT[args.model_name],
                      out_dim=config.OUT_DIM,
                      lora_rank=args.lora_rank,
                      bottleneck_dim=config.BOTTLENECK_DIM,
                      device=device)

    train_dataloaders = []
    test_dataloader = []
    used_stories = train_stories 
    used_stories_sub = {}
    subj_list = config.TRAIN_SUBJECTS

    for s in subj_list:
        if s <= 3:
            used_stories_sub[s] = train_stories 
        else:
            used_stories_sub[s] = train_stories
    num_training_steps = 3 * len(subj_list) * num_epochs * len(train_stories)  # Total steps (adjust per dataset size)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup


    subj_train_dataloaders = {}
    subj_test_dataloaders = {}
    for s in subj_list:
        train_dataloaders = []
        for t_story in used_stories_sub[s]:

            story_ds = FMRIStory(story_name=t_story, subject=s, sub_nc_mask=config.SUB_NEUR_MASK, **config.WAV_PARAMS) 
            
            story_dl = DataLoader(story_ds, batch_size=args.batch_size, shuffle=False)
            train_dataloaders.append(story_dl)
        subj_train_dataloaders[s] = train_dataloaders

        test_ds = FMRIStory(story_name=val_stories[0], subject=s, sub_nc_mask=config.SUB_NEUR_MASK, **config.WAV_PARAMS)
        test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        test_dataloader.dataset.fetch_data()
        subj_test_dataloaders[s] = test_dataloader


    trainable_params = get_train_params(model, has_bottleneck=False)
    # Prepare optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': trainable_params['model'], 'lr': config.BASE_LR},
        {'params': trainable_params['linear'], 'lr': config.LINEAR_LAYER_LR}
    ])
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_training_steps)
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params['model'] + trainable_params['linear'])}")

    # Wrap the model for multi-GPU training
    model = torch.nn.DataParallel(model) 

    # Loss function
    loss_function = get_loss_function(args.loss_name)
    warmup_loss = evaluate(model, test_dataloader, loss_function, device=device)
    print(f'Warmup Loss: {warmup_loss}')
    
    t = time.time()
    # Training loop
    losses_dict = dict(train=[], test=[])
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = [] #TODO: add more train logs

        for s_idx, t_story in enumerate(tqdm((train_stories))): # bind by the story
            story_loss = []

            for s in subj_list:

                s_train_loader = subj_train_dataloaders[s]            
                trainloader = s_train_loader[s_idx]
                trainloader.dataset.fetch_data()
                for input_wav_tensor, output_signal in (trainloader):
                    
                    optimizer.zero_grad()
                    input_wav_tensor = input_wav_tensor.to(device) 
                    output_signal = output_signal.to(device)
                    predictions = model(input_wav_tensor)

                    loss = loss_function(predictions, output_signal)

                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item()); story_loss.append(loss.item())
                
                trainloader.dataset._clear()
            
            print(f'Epoch {epoch+1}, Subj {s}, Story {t_story}, story Train Loss: {np.mean(story_loss)}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
            story_loss = []
            scheduler.step()
                
        # Evaluate on the test set
        eval_losses = []
        
        for s in subj_list:
            s_test_loader = subj_test_dataloaders[s]
            eval_loss = evaluate(model, s_test_loader, loss_function, device=device, sub=s)
            eval_losses.append(eval_loss)
        
        ev_loss = np.mean(eval_losses)    
            
        print(f'Epoch {epoch+1}, Train Loss: {np.mean(epoch_loss)}, Eval Loss: {ev_loss},') # Learning Rate: {scheduler.get_lr()}')
        losses_dict['train'].append(np.mean(epoch_loss)); losses_dict['test'].append(ev_loss)
        epoch_loss = []
        if epoch % 5 == 0 or epoch <= 5:
            torch.save(model.state_dict(), f'{args.save_dir}/{args.exp_name}/wav2vec_story_subj{args.subject}_{args.save_name}_epoch_{epoch}.pth')
    
    # Save the model
    print(f'Training took {time.time() - t} seconds')
    torch.save(model.state_dict(), f'{args.save_dir}/{args.exp_name}/wav2vec_story_subj{args.subject}_{args.save_name}_epoch_{args.num_epochs}.pth')
    pickle.dump(losses_dict, open(f'{args.save_dir}/logs/{args.exp_name}/wav2vec_story_subj{args.subject}_{args.save_name}_losses.pkl', 'wb'))

