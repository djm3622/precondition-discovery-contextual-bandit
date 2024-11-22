from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os


class CriticDataset(Dataset):
    def __init__(self, files, cache, root_dir):
        
        self.data = []
        self.files = files
        self.root_dir = root_dir
        self.cache = cache
        
        if self.cache:            
            for file in self.files:
                self.data = torch.load(os.path.join(self.root_dir, file), weights_only=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache:
            instance = self.data[idx]
        else:
            instance = torch.load(os.path.join(self.root_dir, self.files[idx]), weights_only=True)
            
        return instance['input'], instance['output']
    
    
def get_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, pin_memory=pin_memory)


# we want to use dataloader because of how fast it can read in, 
# but because batches we saved we have to remove the batch added by dataloader
def postprocess(inp, out):
    return inp.view(inp.shape[1:]), out.view(out.shape[1:])