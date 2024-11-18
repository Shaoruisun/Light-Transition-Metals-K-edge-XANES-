##General imports
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import pandas as pd
from torch_geometric.data import InMemoryDataset
##Torch imports
import torch.nn.functional as F
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from xasmod import nets
from torch_geometric.data import InMemoryDataset



def load2dataloader(dataset_path,batch_size=64, shuffle=True):


    assert os.path.exists(dataset_path), "InMemoryDataset not found"
    data, slices = torch.load(dataset_path)
    dataset = InMemoryDataset(transform=None)
    dataset.data, dataset.slices = data, slices
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def model_setup(
    device,
    model_name,
    dataset,
    other_model_params
):
    model = getattr(nets, model_name)(
        data=dataset, **(other_model_params if other_model_params is not None else {})
    ).to(device)
    
    print("Network construction is finished!")
    return model


def load_model(device, model_path, load_mode, model=None):

    assert os.path.exists(model_path), "Saved model not found"
    saved = torch.load(model_path, map_location=torch.device(device))
    if load_mode == "full_model":
        model = saved["full_model"]
        model = model.to(device)
    else:
        assert model != None, "This load mode need a model instance"
        model.load_state_dict(saved["state_dict"])
        
    
    return model


def model_summary(model):
    model_params_list = list(model.named_parameters())
    print(model)
    print("-"*72)
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("-"*72)
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("-"*72)
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
    print()


def train(device, model, optimizer, loader, loss_method, val_method):
    model.train()
    trian_all = 0
    val_all = 0
    count = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        xas_out = model(data)
        sp_loss = getattr(F, loss_method)(xas_out, data.y)
        trian_all += sp_loss.detach() * xas_out.size(0)
        sp_loss.backward()
        optimizer.step()
        sp_val = getattr(F, val_method)(xas_out, data.y)
        val_all += sp_val.detach() * xas_out.size(0)
        

        count = count + xas_out.size(0)

    trian_all_avg = trian_all / count
    val_all_avg = val_all / count
    return trian_all_avg, val_all_avg


def evaluate(device, model, val_loader, loss_method, val_method):
    model.eval()
    trian_all = 0
    val_all = 0
    count = 0
    for data in val_loader:
        with torch.no_grad():
            data = data.to(device)
            xas_out = model(data)
            sp_loss = getattr(F, loss_method)(xas_out, data.y)
            trian_all += sp_loss.detach() * xas_out.size(0)

            sp_val = getattr(F, val_method)(xas_out, data.y)
            val_all += sp_val.detach() * xas_out.size(0)
            count = count + xas_out.size(0)

        trian_all_avg = trian_all / count
        val_all_avg = val_all / count
        return trian_all_avg, val_all_avg


def predict(device, model, test_loader, loss_method, val_method, predict_path):

    model.eval()
    trian_all = 0
    val_all = 0
    count = 0
    for i,data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            xas_out = model(data)
            predict_data = xas_out.reshape([-1,111]).cpu()
            if i == 0:
                id_list = data.data_id
                predict_array = np.array(predict_data)
            else:
                id_list = id_list + data.data_id
                predict_array = np.concatenate((predict_array,np.array(predict_data)),axis=0)
            sp_loss = getattr(F, loss_method)(xas_out, data.y)
            trian_all += sp_loss.detach() * xas_out.size(0)
            sp_val = getattr(F, val_method)(xas_out, data.y)
            val_all += sp_val.detach() * xas_out.size(0)
            count = count + xas_out.size(0)

    trian_all_avg = trian_all / count
    val_all_avg = val_all / count
    predict_pd = pd.DataFrame(predict_array)
    predict_pd.index = id_list
    predict_pd.to_pickle(os.path.join(predict_path, "val_"+str(val_all_avg.cpu())+".pk"))

    print("Testset loss: ",trian_all_avg,"Testset score: ",val_all_avg)
    
    return None


def trainer(
    device,
    model,
    optimizer,
    loss_method, 
    val_method,
    train_loader,
    val_loader,
    epochs,
    verbosity,
    save_path
):

    train_loss_list = []
    train_error_list = []
    val_loss_list = []
    val_error_list = []

    train_loss = train_error = val_loss = val_error = epoch_time = float(1e10)
    best_train_loss = best_val_loss = float(1e10)
    model_best = model
   
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_error = train(device, model, optimizer, train_loader, loss_method, val_method)
        train_loss_list.append(train_loss.item())
        train_error_list.append(train_error.item())
        if val_loader != None:
                val_loss, val_error = evaluate(device, model, val_loader, loss_method, val_method)
                val_loss_list.append(val_loss.item())
                val_error_list.append(val_error.item())
        epoch_time = time.time() - train_start
        train_start = time.time()
        model_path = os.path.join(save_path, "temp_model.pth")    
        if val_loader != None:
            if  val_loss < best_val_loss:
                model_best = copy.deepcopy(model)
                torch.save(
                            {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "full_model": model,
                            },
                            model_path,
                            )
            best_val_loss = min(val_loss, best_val_loss)
        elif val_loader == None:
            if train_loss < best_train_loss:
                model_best = copy.deepcopy(model)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "full_model": model,
                    },
                    model_path,
                )
            best_train_loss = min(train_loss, best_train_loss)
        if epoch % verbosity == 0:
            print(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Loss: {:.5f}, Val Loss: {:.5f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                    epoch, lr, train_loss, val_loss, train_error, val_error, epoch_time
                    )
                )


    loss_path = os.path.join(save_path, "train_loss_results.csv")
    table = pd.DataFrame([train_loss_list,train_error_list,val_loss_list,val_error_list]).T
    table.columns = ["train_loss","train_error","val_loss","val_error"]
    table.to_csv(path_or_buf=loss_path)

    return model_best



def f_r2_score(y_pred, y_true):

    SS_res = torch.sum(torch.square(y_true - y_pred))
    SS_tot = torch.sum(torch.square(y_true - torch.mean(y_true)))
    r2 = 1 - (SS_res / SS_tot)
    
    return r2


def regular_process(parameters):


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = parameters["batch_size"]
    data_path = os.path.join(parameters["data_path"],"Processed")
    train_dataset_path = os.path.join(data_path, "train_dataset.pt")
    val_dataset_path = os.path.join(data_path,"val_dataset.pt")
    test_dataset_path = os.path.join(data_path, "test_dataset.pt")
    train_loader = load2dataloader(dataset_path=train_dataset_path,batch_size=batch_size)
    val_loader = load2dataloader(dataset_path=val_dataset_path,batch_size=batch_size)
    test_loader = load2dataloader(dataset_path=test_dataset_path,batch_size=batch_size)
    for data in train_loader:
        data_samples = data
        break


    model = model_setup(
        device = device,
        model_name ="Three_Sections_GNN",
        dataset = data_samples,
        other_model_params = parameters["model_set"]
    )

    model_summary(model)
    optimizer = getattr(torch.optim, parameters["optimizer"]["name"])(
        model.parameters(),
        lr=parameters["optimizer"]["lr"],
        **parameters["optimizer"]["optimizer_args"]
    )
    
    F.f_r2_score = f_r2_score
    model = trainer(
        device,
        model,
        optimizer = optimizer,
        loss_method = parameters["loss_method"],
        val_method = parameters["score_method"],
        train_loader = train_loader,
        val_loader = val_loader,
        epochs= parameters["epochs"],
        verbosity = parameters["verbosity"],
        save_path= parameters["save_path"],
        )
    print("Model training finished!", "\n")
    predict(
        device = device, 
        model = model, 
        test_loader = test_loader, 
        loss_method = parameters["loss_method"],
        val_method = parameters["score_method"],
        predict_path= parameters["save_path"]
            )
    print("Model predicting finished!", "\n")


def only_test(parameters):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = parameters["batch_size"]
    dataset_path = os.path.join(parameters["data_path"], "test_dataset.pt")
    predict_loader = load2dataloader(dataset_path=dataset_path,batch_size=batch_size)
    model =  load_model(device, parameters["model_path"], load_mode="full_model")    
    model_summary(model)
    F.f_r2_score = f_r2_score
    predict(
        device = device, 
        model = model, 
        test_loader = predict_loader, 
        loss_method = parameters["loss_method"],
        val_method = parameters["score_method"],
        predict_path= parameters["save_path"]
            )
    print("Model testing finished!", "\n")



def only_predict(parameters):


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = parameters["batch_size"]
    dataset_path = os.path.join(parameters["data_path"], "predict_dataset.pt")
    predict_loader = load2dataloader(dataset_path=dataset_path,batch_size=batch_size)
    model =  load_model(device, parameters["model_path"], load_mode="full_model")    
    model.eval()
    for i,data in enumerate(predict_loader):
        data = data.to(device)
        with torch.no_grad():
            xas_out = model(data)
            predict_data = xas_out.reshape([-1,111]).cpu()
            if i == 0:
                id_list = data.data_id
                predict_array = np.array(predict_data)
            else:
                id_list = id_list + data.data_id
                predict_array = np.concatenate((predict_array,np.array(predict_data)),axis=0)


    predict_pd = pd.DataFrame(predict_array)
    predict_pd.index = id_list
    predict_pd.to_pickle(os.path.join(parameters["data_path"], "predict_result.pk"))

    print("Model predicting finished!", "\n")