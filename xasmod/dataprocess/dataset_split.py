import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Subset, ConcatDataset,TensorDataset


def split_inmemorydataset(
                            dataset,
                            seed, 
                            split_ratio=(0.8, 0.1, 0.1), 
                            save_path=None, 
                            verbose=True
                        ):
 

    dataset_length = len(dataset)

    train_idx = int(dataset_length * split_ratio[0])
    test_idx = int(dataset_length * (split_ratio[0] + split_ratio[1]))

    random_seed = np.random.seed(seed)
    perm = list(np.random.permutation(dataset_length))


    train_data_list = []
    val_data_list = []
    test_data_list = []
    for index,i in enumerate(perm):
        if index < train_idx:
            train_data_list.append(dataset[i])
        elif index <= test_idx:
            test_data_list.append(dataset[i])
        else:
            val_data_list.append(dataset[i])

    train_data, train_slices = InMemoryDataset.collate(train_data_list)
    train_dataset = InMemoryDataset(transform=None)
    train_dataset.data, train_dataset.slices = train_data, train_slices
    val_data, val_slices = InMemoryDataset.collate(val_data_list)
    val_dataset = InMemoryDataset(transform=None)
    val_dataset.data, val_dataset.slices = val_data, val_slices
    test_data, test_slices = InMemoryDataset.collate(test_data_list)
    test_dataset = InMemoryDataset(transform=None)
    test_dataset.data, test_dataset.slices = test_data, test_slices


    if save_path != None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save((train_data, train_slices), os.path.join(save_path,"train_dataset.pt"))
        torch.save((val_data, val_slices), os.path.join(save_path,"val_dataset.pt"))
        torch.save((test_data, test_slices), os.path.join(save_path,"test_dataset.pt"))
    
    if verbose == True:
        print(len(dataset),len(train_dataset),len(val_dataset),len(test_dataset),"\n")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":

    print("OK!")
    pass




