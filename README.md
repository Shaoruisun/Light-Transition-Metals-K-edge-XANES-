# A Highly Interpretive "Family Bucket" Model for Simultaneous Predic tion of Multiple Light Transition Metals K-edge XANES 

## Prerequisites

This code has been tested with Pytorch 2.3.0, CUDA 11.8, Window 10.
The library versions used are:

```
ase                       3.23.0
numpy                     1.26.3
scipy                     1.14.1
torch                     2.3.0+cu118
pandas                    2.2.2
pymatgen                  2024.10.27
matplotlib                3.9.2
torch_geometric           2.3.0
torch_scatter             2.1.2+pt23cu118
```

**Due to size limitations the trained weights file is missing here, you can get the full version at https://zenodo.org/records/14177270.**

## Running

All parameter settings can be configured in the `main.py` **###parameters_set** module. Run the `main.py` to get result.

### predict(refer to "./data/cif")

1. Please change **"run_mode"** to **"only_predict"** when you need to predict data.

2. Change **“data_path”** to a folder containing the files **xxx.cif** and **p_spectrum.pt** .

3.  Without sufficient training, do not overwrite the original model ( **.** **/save/temp_model.pth** ), and do not modify anything else and then run it directly.

4. After that, you will get a pickle file named **"predict_result.pk"**, use the method in **". /data/cif/read_pickle.ipynb”**.

####  note that:

The cif file must be named as **Material ID-Crystal Center Absorption Atomic ID-Absorbing Element-XANES-K.cif.** For example, mp-19704-56-Mo-XANES-K. cif, mp-19704 is the material ID, 56 is the crystal center absorbing atom ID (Mo Mo56 1 0.62216022 0.97246597 0.79706718 1), and the absorbing element is Mo.

```
# generated using pymatgen

data_Mg2Tl2(MoO4)3
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   10.99952216
_cell_length_b   10.99952216
_cell_length_c   10.99952216
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Mg2Tl2(MoO4)3
_chemical_formula_sum   'Mg8 Tl8 Mo12 O48'
_cell_volume   1330.82655161
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  O  O0  1  0.95454406  0.28049932  0.02235565  1
  O  O1  1  0.45454406  0.21950068  0.97764435  1
  O  O2  1  0.04545594  0.78049932  0.47764435  1
  O  O3  1  0.21950068  0.97764435  0.45454406  1
  O  O4  1  0.47764435  0.04545594  0.78049932  1
  O  O5  1  0.97764435  0.45454406  0.21950068  1
  O  O6  1  0.28049932  0.02235565  0.95454406  1
  O  O7  1  0.02235565  0.95454406  0.28049932  1
  O  O8  1  0.78049932  0.47764435  0.04545594  1
  O  O9  1  0.71950068  0.52235565  0.54545594  1
  O  O10  1  0.52235565  0.54545594  0.71950068  1
  O  O11  1  0.54545594  0.71950068  0.52235565  1
  O  O12  1  0.18026654  0.24512005  0.89737928  1
  O  O13  1  0.68026654  0.25487995  0.10262072  1
  O  O14  1  0.81973346  0.74512005  0.60262072  1
  O  O15  1  0.25487995  0.10262072  0.68026654  1
  O  O16  1  0.60262072  0.81973346  0.74512005  1
  O  O17  1  0.10262072  0.68026654  0.25487995  1
  O  O18  1  0.72926266  0.05435661  0.70531489  1
  O  O19  1  0.24512005  0.89737928  0.18026654  1
  O  O20  1  0.74512005  0.60262072  0.81973346  1
  O  O21  1  0.75487995  0.39737928  0.31973346  1
  O  O22  1  0.39737928  0.31973346  0.75487995  1
  O  O23  1  0.31973346  0.75487995  0.39737928  1
  O  O24  1  0.94564339  0.20531489  0.77073734  1
  O  O25  1  0.44564339  0.29468511  0.22926266  1
  O  O26  1  0.05435661  0.70531489  0.72926266  1
  O  O27  1  0.29468511  0.22926266  0.44564339  1
  O  O28  1  0.47284029  0.54755501  0.32388740  1
  O  O29  1  0.32388740  0.47284029  0.54755501  1
  O  O30  1  0.54755501  0.32388740  0.47284029  1
  O  O31  1  0.95244499  0.67611260  0.97284029  1
  O  O32  1  0.82388740  0.02715971  0.45244499  1
  O  O33  1  0.45244499  0.82388740  0.02715971  1
  O  O34  1  0.17611260  0.52715971  0.04755501  1
  O  O35  1  0.67611260  0.97284029  0.95244499  1
  O  O36  1  0.04755501  0.17611260  0.52715971  1
  O  O37  1  0.97284029  0.95244499  0.67611260  1
  O  O38  1  0.52715971  0.04755501  0.17611260  1
  O  O39  1  0.02715971  0.45244499  0.82388740  1
  O  O40  1  0.55435661  0.79468511  0.27073734  1
  O  O41  1  0.27073734  0.55435661  0.79468511  1
  O  O42  1  0.79468511  0.27073734  0.55435661  1
  O  O43  1  0.70531489  0.72926266  0.05435661  1
  O  O44  1  0.77073734  0.94564339  0.20531489  1
  O  O45  1  0.20531489  0.77073734  0.94564339  1
  O  O46  1  0.22926266  0.44564339  0.29468511  1
  O  O47  1  0.89737928  0.18026654  0.24512005  1
  Mg  Mg48  1  0.11432877  0.61432877  0.88567123  1
  Mg  Mg49  1  0.38567123  0.38567123  0.38567123  1
  Mg  Mg50  1  0.88567123  0.11432877  0.61432877  1
  Mg  Mg51  1  0.61432877  0.88567123  0.11432877  1
  Mg  Mg52  1  0.66109452  0.66109452  0.66109452  1
  Mg  Mg53  1  0.83890548  0.33890548  0.16109452  1
  Mg  Mg54  1  0.33890548  0.16109452  0.83890548  1
  Mg  Mg55  1  0.16109452  0.83890548  0.33890548  1
  Mo  Mo56  1  0.62216022  0.97246597  0.79706718  1
  Mo  Mo57  1  0.12216022  0.52753403  0.20293282  1
  Mo  Mo58  1  0.29706718  0.87783978  0.02753403  1
  Mo  Mo59  1  0.87783978  0.02753403  0.29706718  1
  Mo  Mo60  1  0.79706718  0.62216022  0.97246597  1
  Mo  Mo61  1  0.70293282  0.37783978  0.47246597  1
  Mo  Mo62  1  0.20293282  0.12216022  0.52753403  1
  Mo  Mo63  1  0.37783978  0.47246597  0.70293282  1
  Mo  Mo64  1  0.02753403  0.29706718  0.87783978  1
  Mo  Mo65  1  0.52753403  0.20293282  0.12216022  1
  Mo  Mo66  1  0.97246597  0.79706718  0.62216022  1
  Mo  Mo67  1  0.47246597  0.70293282  0.37783978  1
  Tl  Tl68  1  0.16594720  0.16594720  0.16594720  1
  Tl  Tl69  1  0.33405280  0.83405280  0.66594720  1
  Tl  Tl70  1  0.66594720  0.33405280  0.83405280  1
  Tl  Tl71  1  0.83405280  0.66594720  0.33405280  1
  Tl  Tl72  1  0.94019730  0.94019730  0.94019730  1
  Tl  Tl73  1  0.55980270  0.05980270  0.44019730  1
  Tl  Tl74  1  0.44019730  0.55980270  0.05980270  1
  Tl  Tl75  1  0.05980270  0.44019730  0.55980270  1
```

### train(refer to "./data/tiny_data")

1. You need to prepare a torch save file, which stores a list of information about this training data, an example of which is shown below:

```python
[['spectrum_id', 'absorbing_element', 'absorbing_index', 'startpoint', 'y'],
 ['mvc-14970-4-Fe-XANES-K', 'Fe', 4, 7112, [0.0, ..., 0.8974457518246562]],
 ['mp-779789-14-Fe-XANES-K', 'Fe', 14, 7112, [0.0 ..., 1.020088926326072]],
 ...]
```

 where **spectrum_id** corresponds to the unique ID of the spectrum, which consists of five parts: Material ID, Center Absorbing Atomic ID, Absorbing element, Spectrum type(XANES), Edge(K); **absorbing_element** corresponds to the absorbing element of the spectrum; **absorbing_index** corresponds to the center absorbing atomic ID; **startpoint** corresponds to the starting point of the spectrum; **y** corresponds to the discretized spectral intensity after the alignment. 

2. You need to prepare the corresponding crystal structure json file corresponding to the above example: mvc-14970.json, mp-779789.json, ...
3. mvc-14970.json example:

```json
{"@module": "pymatgen.core.structure", "@class": "Structure", "lattice": {"matrix": [[6.10383345, 0.02244246, -2.17141007], [-3.03266364, 5.2965814, -2.17193725], [-0.01258519, -0.02244116, 6.47863863]], "a": 6.478603872860815, "b": 6.478281770511496, "c": 6.47868972021332, "alpha": 109.7054249313089, "beta": 109.6945616301035, "gamma": 109.0168523645664, "volume": 209.302353593443}, "sites": [{"species": [{"element": "Ti", "occu": 1}], "abc": [0.99999645, 3.12e-06, 0.49999923], "xyz": [6.097509734171292, 0.0112383429429282, 1.067905188509783], "label": "Ti", "properties": {"coordination_no": 6, "forces": [0.00074063, -0.00086222, 0.0005739]}}, {"species": [{"element": "Ti", "occu": 1}], "abc": [-4.17e-06, 0.49999909, -4.07e-06], "xyz": [-1.516354462039851, 2.648285877861389, -1.085983961816335], "label": "Ti", "properties": {"coordination_no": 6, "forces": [0.00145792, 0.00034967, 0.00090522]}}, {"species": [{"element": "Ti", "occu": 1}], "abc": [0.5000024, 0.49998852, 0.49999378], "xyz": [1.529341852458749, 2.648230738691447, 1.067625080323183], "label": "Ti", "properties": {"coordination_no": 6, "forces": [-0.0022666, 0.00336438, 4.357e-05]}}, {"species": [{"element": "Ti", "occu": 1}], "abc": [0.50000449, 9.13e-06, 1.554e-05], "xyz": [3.051916247419305, 0.011269339819201, -1.085633936373997], "label": "Ti", "properties": {"coordination_no": 6, "forces": [0.00088037, -0.00234585, -0.00304259]}}, {"species": [{"element": "Fe", "occu": 1}], "abc": [0.49998856, 0.499998, 0.99998197], "xyz": [1.522936179383588, 2.637060324709572, 4.306877344951202], "label": "Fe", "properties": {"coordination_no": 6, "forces": [-0.00082412, 4.801e-05, 0.0002592]}}, {"species": [{"element": "Fe", "occu": 1}], "abc": [2.05e-06, 0.50000352, 0.49999933], "xyz": [-1.522622568685363, 2.637088825009148, 2.153334252702354], "label": "Fe", "properties": {"coordination_no": 6, "forces": [0.00063175, -0.00054818, 0.00091063]}}, {"species": [{"element": "Fe", "occu": 1}], "abc": [0.49998277, 0.99999134, 0.49999913], "xyz": [0.01288159476589405, 5.2965358144453, -0.01827238404351704], "label": "Fe", "properties": {"coordination_no": 6, "forces": [-0.00362744, 0.00291797, -0.00378748]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.49323008, 0.80208261, 0.69114527], "xyz": [0.5694492986863241, 4.243855028143337, 1.664602584388652], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.02254684, 0.01859129, -0.03684685]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.11094311, 0.80209285, 0.30885172], "xyz": [-1.759186513811971, 4.243908915826645, 0.0179403590091635], "label": "O", "properties": {"coordination_no": 6, "forces": [0.04183221, 0.01778234, 0.00947672]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.88905981, 0.19790823, 0.69114969], "xyz": [4.817785663983796, 1.052679538371214, 2.117351399661671], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.04211127, -0.01843028, -0.0090728]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.18828425, 0.31182005, 0.50009899], "xyz": [0.1973165345922221, 1.644583037275897, 2.153864737073724], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.01840653, -0.02200521, -0.03811281]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.31388496, 0.49998484, 0.18611259], "xyz": [0.3972734111741523, 2.651078172075173, -0.5617524462934855], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.03500486, -0.0093635, 0.05003488]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.68612133, 0.50002019, 0.81388909], "xyz": [2.661334326497019, 2.645531273191193, 2.697030057392676], "label": "O", "properties": {"coordination_no": 6, "forces": [0.03583356, 0.00883495, -0.04959238]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.81172898, 0.31181966, 0.12353489], "xyz": [4.007459645276939, 1.667023140242742, -1.639511305612363], "label": "O", "properties": {"coordination_no": 6, "forces": [0.0426962, -0.02172003, 0.00544581]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.68613921, 0.87219665, 0.18604514], "xyz": [1.540658980534291, 4.630884136553204, -2.178926752548299], "label": "O", "properties": {"coordination_no": 6, "forces": [0.00801574, 0.06104532, -0.00998741]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.50677245, 0.1979106, 0.30886292], "xyz": [2.489171252725714, 1.05269158105528, 0.4707510394481779], "label": "O", "properties": {"coordination_no": 6, "forces": [0.02311117, -0.0189559, 0.03759082]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.18827789, 0.68817992, 0.87646457], "xyz": [-0.9488318114264065, 3.629557501490996, 3.774785111773967], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.04228411, 0.02130412, -0.00435952]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.81172333, 0.68819005, 0.4999107], "xyz": [2.861283600612073, 3.65205311085525, -0.01854904491995429], "label": "O", "properties": {"coordination_no": 6, "forces": [0.01922613, 0.02108773, 0.03887154]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.31386407, 0.12780165, 0.81395252], "xyz": [1.517950845016957, 0.6656896853819989, 4.314199472602199], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.00735392, -0.0610946, 0.01068952]}}]}
```

### test(refer to "./data/test")

1. Please change **"run_mode"** to **"only_test"** when you need to test the data.

2. Change **“data_path”** to a folder containing the file **test_dataset.pt** (you can refer to or copy the **test_dataset.pt** generated during the **train** process).



## **运行** 

所有参数设置均可在`main.py` **###parameters_set**模块中配置。运行`main.py`获得结果。

### **预测（请参阅“./data/cif”）** 

1. 需要预测数据时，请将 **“run_mode ”**更改为**“only_predict**”。

2. 将**“data_path**”更改为包含**xxx.cif**和**p_spectrum.pt**文件的文件夹。

3. 在没有充分训练的情况下，不要覆盖原始模型 (**.** **/save/temp_model.pth**) ，也不要修改任何其他内容，然后直接运行。

4. 之后会得到一个名为**“predict_result.pk ”**的 pickle 文件，使用方法请参见**”./data/cif/read_pickle.ipynb" **。

####  **注意** 

cif 文件必须以**材料 ID-晶体中心吸收原子 ID-吸收元素-XANES-K.cif**命名。例如，mp-19704-56-Mo-XANES-K.cif，mp-19704 是材料 ID，56 是晶体中心吸收原子 ID（Mo Mo56 1 0.62216022 0.97246597 0.79706718 1），Mo是吸收元素。

```
# generated using pymatgen

data_Mg2Tl2(MoO4)3
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   10.99952216
_cell_length_b   10.99952216
_cell_length_c   10.99952216
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Mg2Tl2(MoO4)3
_chemical_formula_sum   'Mg8 Tl8 Mo12 O48'
_cell_volume   1330.82655161
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  O  O0  1  0.95454406  0.28049932  0.02235565  1
  O  O1  1  0.45454406  0.21950068  0.97764435  1
  O  O2  1  0.04545594  0.78049932  0.47764435  1
  O  O3  1  0.21950068  0.97764435  0.45454406  1
  O  O4  1  0.47764435  0.04545594  0.78049932  1
  O  O5  1  0.97764435  0.45454406  0.21950068  1
  O  O6  1  0.28049932  0.02235565  0.95454406  1
  O  O7  1  0.02235565  0.95454406  0.28049932  1
  O  O8  1  0.78049932  0.47764435  0.04545594  1
  O  O9  1  0.71950068  0.52235565  0.54545594  1
  O  O10  1  0.52235565  0.54545594  0.71950068  1
  O  O11  1  0.54545594  0.71950068  0.52235565  1
  O  O12  1  0.18026654  0.24512005  0.89737928  1
  O  O13  1  0.68026654  0.25487995  0.10262072  1
  O  O14  1  0.81973346  0.74512005  0.60262072  1
  O  O15  1  0.25487995  0.10262072  0.68026654  1
  O  O16  1  0.60262072  0.81973346  0.74512005  1
  O  O17  1  0.10262072  0.68026654  0.25487995  1
  O  O18  1  0.72926266  0.05435661  0.70531489  1
  O  O19  1  0.24512005  0.89737928  0.18026654  1
  O  O20  1  0.74512005  0.60262072  0.81973346  1
  O  O21  1  0.75487995  0.39737928  0.31973346  1
  O  O22  1  0.39737928  0.31973346  0.75487995  1
  O  O23  1  0.31973346  0.75487995  0.39737928  1
  O  O24  1  0.94564339  0.20531489  0.77073734  1
  O  O25  1  0.44564339  0.29468511  0.22926266  1
  O  O26  1  0.05435661  0.70531489  0.72926266  1
  O  O27  1  0.29468511  0.22926266  0.44564339  1
  O  O28  1  0.47284029  0.54755501  0.32388740  1
  O  O29  1  0.32388740  0.47284029  0.54755501  1
  O  O30  1  0.54755501  0.32388740  0.47284029  1
  O  O31  1  0.95244499  0.67611260  0.97284029  1
  O  O32  1  0.82388740  0.02715971  0.45244499  1
  O  O33  1  0.45244499  0.82388740  0.02715971  1
  O  O34  1  0.17611260  0.52715971  0.04755501  1
  O  O35  1  0.67611260  0.97284029  0.95244499  1
  O  O36  1  0.04755501  0.17611260  0.52715971  1
  O  O37  1  0.97284029  0.95244499  0.67611260  1
  O  O38  1  0.52715971  0.04755501  0.17611260  1
  O  O39  1  0.02715971  0.45244499  0.82388740  1
  O  O40  1  0.55435661  0.79468511  0.27073734  1
  O  O41  1  0.27073734  0.55435661  0.79468511  1
  O  O42  1  0.79468511  0.27073734  0.55435661  1
  O  O43  1  0.70531489  0.72926266  0.05435661  1
  O  O44  1  0.77073734  0.94564339  0.20531489  1
  O  O45  1  0.20531489  0.77073734  0.94564339  1
  O  O46  1  0.22926266  0.44564339  0.29468511  1
  O  O47  1  0.89737928  0.18026654  0.24512005  1
  Mg  Mg48  1  0.11432877  0.61432877  0.88567123  1
  Mg  Mg49  1  0.38567123  0.38567123  0.38567123  1
  Mg  Mg50  1  0.88567123  0.11432877  0.61432877  1
  Mg  Mg51  1  0.61432877  0.88567123  0.11432877  1
  Mg  Mg52  1  0.66109452  0.66109452  0.66109452  1
  Mg  Mg53  1  0.83890548  0.33890548  0.16109452  1
  Mg  Mg54  1  0.33890548  0.16109452  0.83890548  1
  Mg  Mg55  1  0.16109452  0.83890548  0.33890548  1
  Mo  Mo56  1  0.62216022  0.97246597  0.79706718  1
  Mo  Mo57  1  0.12216022  0.52753403  0.20293282  1
  Mo  Mo58  1  0.29706718  0.87783978  0.02753403  1
  Mo  Mo59  1  0.87783978  0.02753403  0.29706718  1
  Mo  Mo60  1  0.79706718  0.62216022  0.97246597  1
  Mo  Mo61  1  0.70293282  0.37783978  0.47246597  1
  Mo  Mo62  1  0.20293282  0.12216022  0.52753403  1
  Mo  Mo63  1  0.37783978  0.47246597  0.70293282  1
  Mo  Mo64  1  0.02753403  0.29706718  0.87783978  1
  Mo  Mo65  1  0.52753403  0.20293282  0.12216022  1
  Mo  Mo66  1  0.97246597  0.79706718  0.62216022  1
  Mo  Mo67  1  0.47246597  0.70293282  0.37783978  1
  Tl  Tl68  1  0.16594720  0.16594720  0.16594720  1
  Tl  Tl69  1  0.33405280  0.83405280  0.66594720  1
  Tl  Tl70  1  0.66594720  0.33405280  0.83405280  1
  Tl  Tl71  1  0.83405280  0.66594720  0.33405280  1
  Tl  Tl72  1  0.94019730  0.94019730  0.94019730  1
  Tl  Tl73  1  0.55980270  0.05980270  0.44019730  1
  Tl  Tl74  1  0.44019730  0.55980270  0.05980270  1
  Tl  Tl75  1  0.05980270  0.44019730  0.55980270  1
```

### **训练（请参阅“./data/tiny_data”）** 

1. 您需要准备一个torch save文件，其中存储有关训练数据的信息列表，示例如下：

```python
[['spectrum_id', 'absorbing_element', 'absorbing_index', 'startpoint', 'y'],
 ['mvc-14970-4-Fe-XANES-K', 'Fe', 4, 7112, [0.0, ..., 0.8974457518246562]],
 ['mp-779789-14-Fe-XANES-K'，'Fe'，14，7112，[0.0 ..., 1.020088926326072]], 
 ...]
```

其中**spectrum_id**是光谱的唯一 ID，由材料 ID、中心吸收原子 ID、吸收元素、光谱类型（XANES）、边（K）五部分组成； **absorbing_element**对应光谱的吸收元素；**absorbing_index**对应中心吸收原子 ID；**startpoint**对应光谱的起点；**y**对应对其离散后的光谱强度。

2. 您还需要准备与上述示例相对应的晶体结构 json 文件：mvc-14970.json、mp-779789.json、。。。
3.  mvc-14970.json 示例：

```json
{"@module": "pymatgen.core.structure", "@class": "Structure", "lattice": {"matrix": [[6.10383345, 0.02244246, -2.17141007], [-3.03266364, 5.2965814, -2.17193725], [-0.01258519, -0.02244116, 6.47863863]], "a": 6.478603872860815, "b": 6.478281770511496, "c": 6.47868972021332, "alpha": 109.7054249313089, "beta": 109.6945616301035, "gamma": 109.0168523645664, "volume": 209.302353593443}, "sites": [{"species": [{"element": "Ti", "occu": 1}], "abc": [0.99999645, 3.12e-06, 0.49999923], "xyz": [6.097509734171292, 0.0112383429429282, 1.067905188509783], "label": "Ti", "properties": {"coordination_no": 6, "forces": [0.00074063, -0.00086222, 0.0005739]}}, {"species": [{"element": "Ti", "occu": 1}], "abc": [-4.17e-06, 0.49999909, -4.07e-06], "xyz": [-1.516354462039851, 2.648285877861389, -1.085983961816335], "label": "Ti", "properties": {"coordination_no": 6, "forces": [0.00145792, 0.00034967, 0.00090522]}}, {"species": [{"element": "Ti", "occu": 1}], "abc": [0.5000024, 0.49998852, 0.49999378], "xyz": [1.529341852458749, 2.648230738691447, 1.067625080323183], "label": "Ti", "properties": {"coordination_no": 6, "forces": [-0.0022666, 0.00336438, 4.357e-05]}}, {"species": [{"element": "Ti", "occu": 1}], "abc": [0.50000449, 9.13e-06, 1.554e-05], "xyz": [3.051916247419305, 0.011269339819201, -1.085633936373997], "label": "Ti", "properties": {"coordination_no": 6, "forces": [0.00088037, -0.00234585, -0.00304259]}}, {"species": [{"element": "Fe", "occu": 1}], "abc": [0.49998856, 0.499998, 0.99998197], "xyz": [1.522936179383588, 2.637060324709572, 4.306877344951202], "label": "Fe", "properties": {"coordination_no": 6, "forces": [-0.00082412, 4.801e-05, 0.0002592]}}, {"species": [{"element": "Fe", "occu": 1}], "abc": [2.05e-06, 0.50000352, 0.49999933], "xyz": [-1.522622568685363, 2.637088825009148, 2.153334252702354], "label": "Fe", "properties": {"coordination_no": 6, "forces": [0.00063175, -0.00054818, 0.00091063]}}, {"species": [{"element": "Fe", "occu": 1}], "abc": [0.49998277, 0.99999134, 0.49999913], "xyz": [0.01288159476589405, 5.2965358144453, -0.01827238404351704], "label": "Fe", "properties": {"coordination_no": 6, "forces": [-0.00362744, 0.00291797, -0.00378748]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.49323008, 0.80208261, 0.69114527], "xyz": [0.5694492986863241, 4.243855028143337, 1.664602584388652], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.02254684, 0.01859129, -0.03684685]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.11094311, 0.80209285, 0.30885172], "xyz": [-1.759186513811971, 4.243908915826645, 0.0179403590091635], "label": "O", "properties": {"coordination_no": 6, "forces": [0.04183221, 0.01778234, 0.00947672]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.88905981, 0.19790823, 0.69114969], "xyz": [4.817785663983796, 1.052679538371214, 2.117351399661671], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.04211127, -0.01843028, -0.0090728]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.18828425, 0.31182005, 0.50009899], "xyz": [0.1973165345922221, 1.644583037275897, 2.153864737073724], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.01840653, -0.02200521, -0.03811281]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.31388496, 0.49998484, 0.18611259], "xyz": [0.3972734111741523, 2.651078172075173, -0.5617524462934855], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.03500486, -0.0093635, 0.05003488]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.68612133, 0.50002019, 0.81388909], "xyz": [2.661334326497019, 2.645531273191193, 2.697030057392676], "label": "O", "properties": {"coordination_no": 6, "forces": [0.03583356, 0.00883495, -0.04959238]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.81172898, 0.31181966, 0.12353489], "xyz": [4.007459645276939, 1.667023140242742, -1.639511305612363], "label": "O", "properties": {"coordination_no": 6, "forces": [0.0426962, -0.02172003, 0.00544581]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.68613921, 0.87219665, 0.18604514], "xyz": [1.540658980534291, 4.630884136553204, -2.178926752548299], "label": "O", "properties": {"coordination_no": 6, "forces": [0.00801574, 0.06104532, -0.00998741]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.50677245, 0.1979106, 0.30886292], "xyz": [2.489171252725714, 1.05269158105528, 0.4707510394481779], "label": "O", "properties": {"coordination_no": 6, "forces": [0.02311117, -0.0189559, 0.03759082]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.18827789, 0.68817992, 0.87646457], "xyz": [-0.9488318114264065, 3.629557501490996, 3.774785111773967], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.04228411, 0.02130412, -0.00435952]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.81172333, 0.68819005, 0.4999107], "xyz": [2.861283600612073, 3.65205311085525, -0.01854904491995429], "label": "O", "properties": {"coordination_no": 6, "forces": [0.01922613, 0.02108773, 0.03887154]}}, {"species": [{"element": "O", "occu": 1}], "abc": [0.31386407, 0.12780165, 0.81395252], "xyz": [1.517950845016957, 0.6656896853819989, 4.314199472602199], "label": "O", "properties": {"coordination_no": 6, "forces": [-0.00735392, -0.0610946, 0.01068952]}}]}
```

### **测试（请参阅“./data/test”）** 

1. 需要测试数据时，请将**“run_mode**”改为**“only_test**”。

2. 将**“data_** **path**”更改为包含**test_dataset.pt**文件的文件夹（您可以参考或复制在**训练**过程中生成的**test_dataset.pt**）。
