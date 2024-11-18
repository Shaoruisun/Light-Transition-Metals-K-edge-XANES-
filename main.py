import os
import argparse
import time
import json
import yaml
import torch
import numpy as np

import xasmod


def get_dataset(parameters):

    import xasmod.dataprocess as datap
    print("\n", os.getcwd(),"\n")
    if not os.path.exists(os.path.join(parameters["data_path"], "p_spectrum.pt")):
        print("Not find p_spectrum.pt!")
    if not os.path.exists(os.path.join(parameters["data_path"],"Processed", "dataset.pt")):
        datap.build_dataset.building(data_path=parameters["data_path"],topology_parameters=parameters["topology"])
    print("Build completed!","\n")
    if parameters["run_mode"] == "regular_process":
        from torch_geometric.data import InMemoryDataset
        data, slices = torch.load(os.path.join(parameters["data_path"],"Processed", "dataset.pt"))
        dataset = InMemoryDataset(transform=None)
        dataset.data, dataset.slices = data, slices
        datap.dataset_split.split_inmemorydataset(dataset=dataset, seed=1, split_ratio=(0.8, 0.1, 0.1),save_path=os.path.join(parameters["data_path"],"Processed"))
        print("Split dataset completed!", "\n")


    return


def run_model():
    start_time = time.time()
    print("Starting...")
    print(
        "GPU is available:",
        torch.cuda.is_available(),
        ", Quantity: ",
        torch.cuda.device_count(),
    )
    parameters = parameters_set()

    if parameters["run_mode"] == "regular_process":
        print("Starting regular training")
        if not os.path.exists(os.path.join(parameters["data_path"],"Processed", "train_dataset.pt")):
            get_dataset(parameters)
        xasmod.training.regular_process(parameters=parameters)

    elif parameters["run_mode"] == "only_test":
        print("Starting testing")
        if  os.path.exists(os.path.join(parameters["data_path"], "test_dataset.pt")):
            xasmod.training.only_test(parameters=parameters)
    
    elif parameters["run_mode"] == "only_predict":
        print("Starting predicting")
        if not os.path.exists(os.path.join(parameters["data_path"], "predict_dataset.pt")):
            xasmod.dataprocess.predict_building(parameters["data_path"],parameters["topology"],verbose=True)
        print("Build completed!","\n")
        xasmod.training.only_predict(parameters=parameters)

    print("--- Total Time:  %s （s） ---" % (time.time() - start_time))
################################################################################
###parameters_set
def parameters_set():

    parameters = dict()

    parameters["run_mode"] = "only_predict"                             # regular_process/only_test/only_predict

    parameters["data_path"] = "./data/cif"                              # "./data/tinydata" ,"./data/test", "./data/cif"

    parameters["topology"] = {
                                "topology_mode": "atomic environment",  # full crystal/atomic environment
                                "topology_radius":8,
                                "topology_atoms_num":20
                             }
    
    parameters["batch_size"] = 160
    
    parameters["model_set"] = {
                                "dims":[[600,500,400],[300],[200,200]], # SGCN,MHGA,MLP(n-1)
                                "heads": 3,
                                "batch_norm":True,
                                "batch_track_stats":"True",
                                "dropout_rate":0.05,
                                "edge_features": 65
                              }
    
    parameters["optimizer"] = {
                                "lr": 0.0001,
                                "name": "AdamW",
                                "optimizer_args": {"weight_decay":0.001}
                              }
    
    parameters["loss_method"] = "l1_loss"

    parameters["score_method"] = "f_r2_score"

    parameters["epochs"] = 800

    parameters["verbosity"] = 1

    parameters["save_path"] = "./save/"
    if  not os.path.isdir(parameters["save_path"]):
        os.mkdir(parameters["save_path"])


    parameters["model_path"] = "./save/temp_model.pth"
    
    return parameters

################################################################################
if __name__ == "__main__":
    run_model()
