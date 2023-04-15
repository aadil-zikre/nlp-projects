import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
tqdm.pandas()

from torch.utils.data import Dataset, DataLoader

def custom_collate(data): 
    tokenizer = data[0][2]
    X = [i[0] for i in data]
    Y = torch.tensor([i[1] for i in data])
    return tokenizer(X, padding='max_length', truncation='longest_first', return_tensors='pt', return_attention_mask=True, return_token_type_ids=True), Y

def custom_collate_test(data): 
    tokenizer = data[0][1]
    X = [i[0] for i in data]
    # Y = torch.tensor([i[1] for i in data])
    return tokenizer(X, padding='max_length', truncation='longest_first', return_tensors='pt', return_attention_mask=True, return_token_type_ids=True)


class TF_Dataset(Dataset):
    def __init__(self, df, col, tokenizer, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.col = col
        self.X = list(df[self.col])
        self.is_test = kwargs.get('is_test', False)
        if not self.is_test:
            self.Y = list(df['sentiment_id']) 
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # x_encoded = self.tokenizer(self.X[idx], padding='longest', truncation='longest_first', return_tensors='pt', return_attention_mask=True, return_token_type_ids=True)
        if self.is_test:
            return self.X[idx], self.tokenizer
        return self.X[idx], self.Y[idx], self.tokenizer
    

class BaseProcessor:
    def __init__(self, df, col, model_string, MODEL_MAX_LENGTH, DEVICE, BATCH_SIZE, random_state, **kwargs):
        self.df = df
        self.col = col
        self.MODEL_MAX_LENGTH = MODEL_MAX_LENGTH
        self.device = DEVICE
        self.model_string = model_string
        self.tf_dataset = TF_Dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_string,
                                          model_max_length = MODEL_MAX_LENGTH,
                                          padding_side = "right",
                                          truncation_side = "right")
        self.model = AutoModel.from_pretrained(self.model_string)
        self.BATCH_SIZE = BATCH_SIZE
        self.random_state = random_state
        self.is_test = kwargs.get('is_test', False)

    
    def _create_dataloader(self):
        
        dataset = self.tf_dataset(self.df, self.col, self.tokenizer, is_test = self.is_test)
        if self.is_test:
            dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, collate_fn=custom_collate_test)
        else:
            dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, collate_fn=custom_collate)

        return dataloader
    
    def get_training_XY(self, test_size = 0.25):
        self.model.to(self.device)
        dataloader = self._create_dataloader()
        self.model.eval()
        X = []
        Y = []
        for batch in tqdm(dataloader):
            batch = tuple(b.to(self.device) for b in batch)
            with torch.no_grad():
                out = self.model(**batch[0])['last_hidden_state'].cpu().numpy()
                out_y = batch[1].cpu().numpy()
            X.append(out)
            Y.append(out_y)
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Y = pd.get_dummies(Y).values
        if not test_size:
            return X,Y
        else:
            return train_test_split(X,Y, test_size=test_size, random_state = self.random_state)
        
    def get_training_X(self):
        self.model.to(self.device)
        dataloader = self._create_dataloader()
        self.model.eval()
        X = []
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            with torch.no_grad():
                out = self.model(**batch)['last_hidden_state'].cpu().numpy()
            X.append(out)
        X = np.concatenate(X)
        return X
    
    
class BertProcessor(BaseProcessor):
    def __init__(self, df, col, MODEL_MAX_LENGTH, DEVICE, BATCH_SIZE, random_state, **kwargs):
        model_string = 'bert-base-uncased'
        super().__init__(df, col, model_string, MODEL_MAX_LENGTH, DEVICE, BATCH_SIZE, random_state, **kwargs)
        

class XLNetProcessor(BaseProcessor):
    def __init__(self, df, col, MODEL_MAX_LENGTH, DEVICE, BATCH_SIZE, random_state, **kwargs):
        model_string = 'xlnet-base-cased'
        super().__init__(df, col, model_string, MODEL_MAX_LENGTH, DEVICE, BATCH_SIZE, random_state, **kwargs)
    
