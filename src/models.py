import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Model, Wav2Vec2Processor, HubertModel, Wav2Vec2FeatureExtractor
    
class Wav2VecLinear(nn.Module):
    def __init__(self, out_dim,
                 wav2vec_model_name='facebook/wav2vec2-base-960h',
                 pooling_strategy='mean',
                 sampling_rate=16000,):
        super(Wav2VecLinear, self).__init__()
        print(f'loading wav2vec model with {wav2vec_model_name}')
        self.wav2vec = Wav2Vec2Model.from_pretrained(f'{wav2vec_model_name}')
        self.processor = Wav2Vec2Processor.from_pretrained(f"{wav2vec_model_name}", return_tensors="pt")
        self.wav2vec.freeze_feature_encoder()
        # print(self.wav2vec.config.hidden_size)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.wav2vec.config.hidden_size, out_dim)
        self.pooling_strategy = pooling_strategy
        self.sampling_rate = sampling_rate
    
    def forward(self, input_wav_tensor):
        # with torch.no_grad():  # Optionally freeze wav2vec to prevent fine-tuning its weights
        tensor_device = input_wav_tensor.device
        bsize = len(input_wav_tensor)
        processed_output = self.processor(input_wav_tensor, return_tensors="pt", sampling_rate=self.sampling_rate).input_values.squeeze().reshape(bsize, -1).to(tensor_device)
        extracted_features = self.wav2vec(processed_output).last_hidden_state
        hidden_states = self.merged_strategy(extracted_features, mode=self.pooling_strategy)

        output = self.linear(hidden_states)
        return output

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs
# Load pre-trained wav2vec model
from peft import LoraConfig, get_peft_model

class Wav2VecLoRA(nn.Module):
    def __init__(self, 
                 out_dim,
                 bottleneck_dim=200,
                 lora_rank=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 wav2vec_model_name='facebook/wav2vec2-base-960h',
                 pooling_strategy='mean',
                 sampling_rate=16000,):
        super(Wav2VecLoRA, self).__init__()
        print(f'loading wav2vec model with {wav2vec_model_name}')
        self.wav2vec = Wav2Vec2Model.from_pretrained(f'{wav2vec_model_name}')
        self.processor = Wav2Vec2Processor.from_pretrained(f"{wav2vec_model_name}", return_tensors="pt", return_attention_mask=True )
        self.wav2vec.freeze_feature_encoder()
        self.wav2vec.eval()
        lora_config = LoraConfig(
            r=lora_rank,  # LoRA rank
            lora_alpha=lora_alpha,  # Scaling factor
            target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"],  # Target self-attention layers
            lora_dropout=lora_dropout,  # Dropout for regularization
            bias="none"
        )
        self.lora_model = get_peft_model(self.wav2vec, lora_config)
        self.lora_model.print_trainable_parameters()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.wav2vec.config.hidden_size, out_dim)
        
        self.pooling_strategy = pooling_strategy
        self.sampling_rate = sampling_rate
    
    def forward(self, input_wav_tensor):
        tensor_device = input_wav_tensor.device
        bsize = len(input_wav_tensor)
        processed_output = self.processor(input_wav_tensor, return_tensors="pt", sampling_rate=self.sampling_rate).input_values.squeeze().reshape(bsize, -1).to(tensor_device)

        extracted_features = self.lora_model(processed_output).last_hidden_state
        hidden_states = self.merged_strategy(extracted_features, mode=self.pooling_strategy)
        output = self.linear(hidden_states)
        return output
    def get_trainable_parameters(self):
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.lora_model.parameters() + self.bottlneck.parameters() + self.linear.parameters() )}")
        return self.lora_model.parameters() + self.bottlneck.parameters() + self.linear.parameters() 
    def print_trainable_parameters(self):
        ## print # trainable params
        print(f"Number of trainable parameters: {sum(p.numel() for p in list(self.lora_model.parameters()) + list(self.bottleneck.parameters()) + list(self.linear.parameters()) )}")

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

class HubertLinear(nn.Module):
    def __init__(self,
                 out_dim,
                 model_name="facebook/hubert-base-ls960",
                 return_hidden=False,
                 pooling_strategy='mean',
                 sampling_rate=16000,):
        super(HubertLinear, self).__init__()
        self.return_hidden = return_hidden
        print('loading model', model_name)
        self.hubert = HubertModel.from_pretrained(model_name)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, return_tensors="pt")
        self.hubert.feature_extractor._freeze_parameters()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.hubert.config.hidden_size, out_dim)
        self.pooling_strategy = pooling_strategy
        self.sampling_rate = sampling_rate
    
    def forward(self, input_wav_tensor):
        tensor_device = input_wav_tensor.device
        bsize = len(input_wav_tensor)
        processed_output = self.processor(input_wav_tensor, return_tensors="pt", sampling_rate=self.sampling_rate).input_values.squeeze().reshape(bsize, -1).to(tensor_device)
        extracted_features = self.hubert(processed_output, output_hidden_states=self.return_hidden).last_hidden_state
        hidden_states = self.merged_strategy(extracted_features, mode=self.pooling_strategy)

        output = self.linear(hidden_states)
        return output

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

class HuBERTLoRA(nn.Module):
    def __init__(self, 
                 out_dim,
                 bottleneck_dim=200,
                 lora_rank=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 model_name='facebook/hubert-base-ls960',
                 pooling_strategy='mean',
                 sampling_rate=16000,):
        super(HuBERTLoRA, self).__init__()
        print(f'loading HuBERT model with {model_name}')
        self.hubert = HubertModel.from_pretrained(f'{model_name}')
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"{model_name}", return_tensors="pt",)
        self.hubert.feature_extractor._freeze_parameters()
        self.hubert.eval()
        
        lora_config = LoraConfig(
            r=lora_rank,  # LoRA rank
            lora_alpha=lora_alpha,  # Scaling factor
            target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"],  # Target self-attention layers
            lora_dropout=lora_dropout,  # Dropout for regularization
            bias="none"
        )
        self.lora_model = get_peft_model(self.hubert, lora_config)
        self.lora_model.print_trainable_parameters()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.hubert.config.hidden_size, out_dim)
        
        self.pooling_strategy = pooling_strategy
        self.sampling_rate = sampling_rate
    
    def forward(self, input_wav_tensor):
        tensor_device = input_wav_tensor.device
        bsize = len(input_wav_tensor)
        processed_output = self.processor(input_wav_tensor, return_tensors="pt", sampling_rate=self.sampling_rate).input_values.squeeze().reshape(bsize, -1).to(tensor_device)

        extracted_features = self.lora_model(processed_output).last_hidden_state
        hidden_states = self.merged_strategy(extracted_features, mode=self.pooling_strategy)

        output = self.linear(hidden_states)
        return output
    def get_trainable_parameters(self):
        ## print # trainable params
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.lora_model.parameters() +  self.linear.parameters() )}")
        return self.lora_model.parameters() + self.bottlneck.parameters() + self.linear.parameters() 
    def print_trainable_parameters(self):
        ## print # trainable params
        print(f"Number of trainable parameters: {sum(p.numel() for p in list(self.lora_model.parameters()) + list(self.linear.parameters()) )}")
        # return self.lora_model.get_trainable_parameters() + self.bottlneck.parameters() + self.linear.parameters() 

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs
    
