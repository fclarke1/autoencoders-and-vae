from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CustomDataloaders() :
    
    @staticmethod
    def MNIST(batch_size:int=64) -> tuple[DataLoader, DataLoader]:
        train_data = datasets.MNIST(
            root='data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        test_data = datasets.MNIST(
            root='data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        
        return (train_dataloader, test_dataloader)