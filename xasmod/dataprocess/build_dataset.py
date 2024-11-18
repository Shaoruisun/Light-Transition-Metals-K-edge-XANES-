import os
import csv
import ase
import json
import torch
import warnings
import numpy as np
import pandas as pd
from ase import io
from scipy.stats import rankdata
from pymatgen.core.structure import Structure
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, add_self_loops

def load_stru(stru_path):
    warnings.filterwarnings('ignore')
    stru_crystal = Structure.from_file(stru_path)
    return stru_crystal


def structure_to_ase(structure):
    cell = structure.lattice.matrix
    symbols = [str(site.specie) for site in structure]
    positions = structure.cart_coords
    ase_crystal = ase.Atoms(symbols=symbols, positions=positions, cell=cell)

    return ase_crystal


def stru_topology(stru_crystal,radius,max_num_nbr):
    coords = stru_crystal.cart_coords
    all_nbrs = stru_crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []

    for i,nbr in enumerate(all_nbrs):
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(), nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(), nbr)) +
                [[coords[i][0]+radius,coords[i][1],coords[i][2]]] * (max_num_nbr -len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(),
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(),
                                    nbr[:max_num_nbr])))

    nbr_subtract=[]
    nbr_distance=[]

    for i in range(len(nbr_fea)):
        if nbr_fea[i] != []:
            x=nbr_fea[i]-coords[:,np.newaxis,:][i]
            nbr_subtract.append(x)
            nbr_distance.append(np.linalg.norm(x, axis=1).tolist())
        else:
            nbr_subtract.append(np.array([]))
            nbr_distance.append(np.array([]))

    nbr_fea_idx = np.array(nbr_fea_idx) 
    return nbr_fea_idx,nbr_distance,nbr_subtract


def rbf_f(distance,cutoff,n_Gaussian):
    combine_sets=[]
    N=n_Gaussian
    for n in range(1,N+1):
        phi=Phi(distance,cutoff)
        G=gaussian(distance,miuk(n,N,cutoff),betak(N,cutoff))
        combine_sets.append(phi*G)
    combine_sets=np.array(combine_sets, dtype=np.float32).transpose()
    return combine_sets
def Phi(r,cutoff):
    return 1-6*(r/cutoff)**5+15*(r/cutoff)**4-10*(r/cutoff)**3
def gaussian(r,miuk,betak):
    return np.exp(-betak*(np.exp(-r)-miuk)**2)
def miuk(n,K,cutoff):
    return np.exp(-cutoff)+(1-np.exp(-cutoff))/K*n
def betak(K,cutoff):
    return (2/K*(1-np.exp(-cutoff)))**(-2)


def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass



def building(data_path,topology_parameters,verbose=True):
    
    import xasmod.dataprocess.side_information as sd
    electrons_dict = sd.electrons_dict(unfold=True)
    data_file_path = os.path.join(data_path, "p_spectrum.pt")
    assert os.path.exists(data_file_path), ("Data file not found in " + data_file_path)
    xy_list = torch.load(f=data_file_path)
    xy_list_head = xy_list[0]
    xy_list = xy_list[1:]
    data_list = []
    for index,i in enumerate(xy_list):
        data_id,abs_element,abs_index,starting=i[0],i[1],i[2],i[3]
        data = Data()
        data.data_id = data_id
        data.absorb = (abs_element,abs_index)
        data.start = starting
        data.y = torch.tensor(i[4],dtype=torch.float32)
        mid = data_id.split("-", 2)[0]+"-"+data_id.split("-", 2)[1]
        stru_path = os.path.join(data_path, mid+".json")
        stru_crystal = load_stru(stru_path)
        data.stru = stru_crystal
        data.atomic_number = torch.LongTensor(stru_crystal.atomic_numbers)
        data.simple_edge = stru_topology(stru_crystal, topology_parameters["topology_radius"], topology_parameters["topology_atoms_num"])
        index_order_matrix, distance_order_matrix, vector_order_matrix = data.simple_edge
        if topology_parameters["topology_mode"] == "full crystal":
            node_num = len(data.atomic_number)
            edge_sources = np.concatenate([[i] * len(index_order_matrix[i]) for i in range(node_num)])
            edge_targets = np.concatenate(index_order_matrix)
            edge_index = torch.tensor(np.array([edge_sources,edge_targets]),dtype=torch.int64)
        else:
            ai = data.absorb[1]
            index_order_matrix, distance_order_matrix, vector_order_matrix = \
            index_order_matrix[ai], distance_order_matrix[ai], vector_order_matrix[ai]
            data.simple_edge = [index_order_matrix], [distance_order_matrix], [vector_order_matrix]
            edge_sources = np.concatenate([[ai] * len(index_order_matrix)])
            edge_targets = np.concatenate([index_order_matrix])
            edge_index = torch.tensor(np.array([edge_sources,edge_targets]),dtype=torch.int64)
        data.edge_index = edge_index
        data_list.append(data)
        if int(index+1) % 200 == 0:
            print("Graph building progress: ",index+1,"/",len(xy_list))
        
            
    for index in range(0, len(data_list)):
        ase_crystal = structure_to_ase(data_list[index].stru)
        x_matrix = []
        for i in range(len(ase_crystal)):
            element = ase_crystal.get_chemical_symbols()[i]
            x_i = electrons_dict[str(element)]
            x_matrix.append(x_i)
        x_matrix = np.vstack(x_matrix).astype(float)
        data_list[index].x = torch.Tensor(x_matrix)
        _, distance_order_matrix, vector_order_matrix = data_list[index].simple_edge
        distance = np.concatenate(distance_order_matrix)
        vector = np.concatenate(vector_order_matrix)
        rbf = rbf_f(distance, topology_parameters["topology_radius"], 64)
        edge_attr = np.concatenate((distance[:,np.newaxis], rbf), axis=1)
        data_list[index].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        if int(index+1) % 200 == 0:
            print("Feature processing progress: ",index+1,"/",len(data_list))


    Cleanup(data_list, [
                        "stru", "atomic_number", 
                        "edge_descriptor", "absorb","simple_edge"
                        ])


    save_path = os.path.join(data_path, "Processed")
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(save_path, "dataset.pt"))


def predict_building(data_path,topology_parameters,verbose=True):

    import xasmod.dataprocess.side_information as sd
    electrons_dict = sd.electrons_dict(unfold=True)
    
    
    import glob
    files = glob.glob(data_path+"/*-XANES-K.cif")
    print("Collect Files: ", len(files),".")
    data_list = []
    for index,i in enumerate(files):
        data = Data()
        data.data_id = i.split("\\")[-1][:-4]
        print(data.data_id)
        data.absorb = (data.data_id.split("-")[-3],int(data.data_id.split("-")[-4]))
        stru_crystal = load_stru(i)
        data.stru = stru_crystal
        data.atomic_number = torch.LongTensor(stru_crystal.atomic_numbers)
        data.simple_edge = stru_topology(stru_crystal, topology_parameters["topology_radius"], topology_parameters["topology_atoms_num"])
        index_order_matrix, distance_order_matrix, vector_order_matrix = data.simple_edge
        if topology_parameters["topology_mode"] == "full crystal":
            node_num = len(data.atomic_number)
            edge_sources = np.concatenate([[i] * len(index_order_matrix[i]) for i in range(node_num)])
            edge_targets = np.concatenate(index_order_matrix)
            edge_index = torch.tensor(np.array([edge_sources,edge_targets]),dtype=torch.int64)
        else:
            ai = data.absorb[1]
            index_order_matrix, distance_order_matrix, vector_order_matrix = \
            index_order_matrix[ai], distance_order_matrix[ai], vector_order_matrix[ai]
            data.simple_edge = [index_order_matrix], [distance_order_matrix], [vector_order_matrix]
            edge_sources = np.concatenate([[ai] * len(index_order_matrix)])
            edge_targets = np.concatenate([index_order_matrix])
            edge_index = torch.tensor(np.array([edge_sources,edge_targets]),dtype=torch.int64)
        data.edge_index = edge_index
        data_list.append(data)
        if int(index+1) % 200 == 0:
            print("Graph building progress: ",index+1,"/",len(files))
        
            
    for index in range(0, len(data_list)):
        ase_crystal = structure_to_ase(data_list[index].stru)
        x_matrix = []
        for i in range(len(ase_crystal)):
            element = ase_crystal.get_chemical_symbols()[i]
            x_i = electrons_dict[str(element)]
            x_matrix.append(x_i)
        x_matrix = np.vstack(x_matrix).astype(float)
        data_list[index].x = torch.Tensor(x_matrix)
        _, distance_order_matrix, vector_order_matrix = data_list[index].simple_edge
        distance = np.concatenate(distance_order_matrix)
        vector = np.concatenate(vector_order_matrix)
        rbf = rbf_f(distance, topology_parameters["topology_radius"], 64)
        edge_attr = np.concatenate((distance[:,np.newaxis], rbf), axis=1)
        data_list[index].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        if int(index+1) % 200 == 0:
            print("Feature processing progress: ",index+1,"/",len(data_list))


    Cleanup(data_list, [
                        "stru", "atomic_number", 
                        "edge_descriptor", "absorb","simple_edge"
                        ])

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(data_path, "predict_dataset.pt"))
    


if __name__ == "__main__":

    print("OK!")
    pass