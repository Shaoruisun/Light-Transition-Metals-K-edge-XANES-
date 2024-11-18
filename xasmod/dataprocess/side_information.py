import os
import json


def starting_dict(starting_dict_path=None):

    if starting_dict_path == None:
        starting_dict_path = os.path.join("./","xasmod", "information","starting_dict.json")
    assert os.path.exists(starting_dict_path), ("Starting dict file not found in " + starting_dict_path)
    with open(starting_dict_path) as f:
        starting_dict = json.load(f)
    
    return starting_dict


def electrons_dict(unfold=False,electrons_dict_path=None):

    if electrons_dict_path == None:
        electrons_dict_path = os.path.join("./","xasmod", "information","detail_electrons_dict.json")
    assert os.path.exists(electrons_dict_path), ("Detail electrons dict file not found in " + electrons_dict_path)
    with open(electrons_dict_path) as f:
        data = json.load(f)
    del data["record_order"]
    for key in list(data.keys()):
        electrons = []
        for shell in data[key]:
            electrons = electrons + shell
        data[key] = electrons
    
    if unfold == True:
        for key in list(data.keys()):
            electrons = []
            for shell in data[key]:
                if shell == 0:
                    electrons = electrons + [0,0]
                elif shell == 1:
                    electrons = electrons + [1,0]
                elif shell == 2:
                    electrons = electrons + [1,1]  
            data[key] = electrons


    electrons_dict = data
    return electrons_dict


if __name__ == "__main__":

    print("OK!")
    pass


