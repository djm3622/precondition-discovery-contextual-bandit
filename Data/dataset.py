from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch


class SystemDataset(Dataset):
    def __init__(self, generator, norm_feat, epoch_len, batch_size, size=5, perturbation_strength=1e-5, transform=None):
        super().__init__(None, transform)
        
        self.generator = generator
        self.norm_features = norm_feat
        self.epoch_len = batch_size*epoch_len
        self.size = size
        self.perturbation_strength = perturbation_strength

    def len(self):
        return self.epoch_len

    def get(self, idx):
        A = torch.from_numpy(self.generator(self.size, perturbation_strength=self.perturbation_strength)[0].toarray())
        A = self.norm_features(A)
        A = torch.where(torch.abs(A) < 1e-1, torch.zeros_like(A), A)
        b = torch.randn(self.size).unsqueeze(-1)
        
        return A, b
    
    
def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)