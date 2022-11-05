# MechRetro
You can find our preprint in [here](https://www.researchgate.net/publication/364222773_MechRetro_is_a_chemical-mechanism-driven_graph_learning_framework_for_interpretable_retrosynthesis_prediction_and_pathway_planning)
# Overview
MechRetro is a chemical-mechanism-like graph learning framework via self-adaptive joint learning for interpretable retrosynthesis prediction and pathway planning
![image](framework.png)
# Installation
**Download the repository**

```shell
git clone git@github.com:wy-sdu/MechRetro.git
cd MechRetro
```

**Install required packages**

We recommend to use anaconda to get the dependencies. If you don't already have anaconda, install it by following instructions at this link: https://docs.anaconda.com/anaconda/install/.

Then, open the terminal on Linux and enter the  following commands to create a new conda environment and install the  required packages:

```sh
conda create -n mechretro --file requirement.txt
conda activate mechretro
```

If the previous commands don't work, you can install the packages separately:

1. Create a new virtual environment with a 3.7.13 version of python:

   ```sh
   conda create -n mechretro python=3.7.13
   conda activate mechretro
   ```

2. Install `pytroch` with correct CUDA version. To find your suitable version, see https://pytorch.org/get-started/locally/

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
   ```

3. Install `pytorch-lightning`:

   ```shell
   conda install pytorch-lightning -c conda-forge
   ```

4. Install `torch-geometric` and its corresponding dependencies, see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.

   ```shell
   conda install pyg -c pyg
   ```

5. Install rdkit:

   ```shell
    conda install -c conda-forge rdkit
   ```

Other packages can be easily installed by calling `pip install xxx` command.

# Model Training

The `USPTO50K`  datasets can be download [here](https://drive.google.com/file/d/12WnLFJ6LSVj6Z47ZTREEpMXAOKTNAjJe/view?usp=share_link) , and should be put under the folders `/data` .

You can run the training stages by following command for reaction type known conditions:

```shell
sh scripts/train_for_uspto50k.sh
```

And for reaction type unknown:

```shell
sh scripts/train_for_uspto50k_rxn_type_unknown.sh
```

 As for your own datasets, you can put them into the `data/your_datasets/raw` , and set the argument to `data/your_datasets ` .

# Reproduce results

At first, you can download the checkpoints [here](https://drive.google.com/file/d/1GgYO8SjKonlkUKhsthp2R8wo2onc0SMI/view?usp=share_link) for reaction-type-known model and [here](https://drive.google.com/file/d/1Z5IzsCEOgO-_rgLBa0VZZvMyV36-eIOG/view?usp=share_link) for unknown model. Or you can train your own model according to the section **Model Training** . 

Then, put the checkpoints under the `model_saved` folder, and run the following command:

```shell
sh scripts/test_for_uspto50k.sh
```

Similarly, for reaction type unknown conditions, you can:

```shell
sh scripts/test_for_uspto50k_rxn_type_unknown.sh
```

# Interpretable Multi-Step Planing

For multi-step predictions, you can click [here](https://drive.google.com/file/d/1HxDJKe5WyHFet-YOmWP3EpOwT_uAR3yr/view?usp=share_link) to get purchasable molecule set, and put it under the folder `/data/multi-step/retro_data/dataset/`. 

We provide a more robust checkpoint trained on larger scale datasets, which takes about a week on three 3090 GPUs.  You can download the checkpoint [here](https://drive.google.com/file/d/1HxDJKe5WyHFet-YOmWP3EpOwT_uAR3yr/view?usp=share_link). And don't forget put it under the folder `model_saved`.

You can get the reasoning process and their energy scores by modifying ` smi_list` in  `api_for_multisetp.py`:

```python
if __name__ == '__main__':
    planner = RSPlanner(
        gpu=-1,
        use_value_fn=False,
        iterations=500,
        expansion_topk=10,
        viz=True,
        viz_dir='data/multi-step/viz'
    )
    smi_list = [
        "CC(CC1=CC=C2OCOC2=C1)NCC(O)C1=CC=C(O)C(O)=C1"
    ]
    for smi in smi_list:
        result = planner.plan(smi, need_action=True)
        print(result)
```

Then you can get following retrosynthetic routes:

```
{'succ': True, 'time': 48.523579359054565, 'iter': 18, 'routes': "CC(CC1=CC=C2OCOC2=C1)NCC(O)C1=CC=C(O)C(O)=C1>3.2358>NCC(O)c1ccc(O)c(O)c1.CC(=O)Cc1ccc2c(c1)OCO2|NCC(O)c1ccc(O)c(O)c1>7.8090>NC(=O)C(O)c1ccc(O)c(O)c1|CC(=O)Cc1ccc2c(c1)OCO2>10.1133>CC(O)Cc1ccc2c(c1)OCO
2|NC(=O)C(O)c1ccc(O)c(O)c1>9.0024>N.O=C(O)C(O)c1ccc(O)c(O)c1[['Select Leaving Group with Index 6 and Cost 0.43', 'Initial Cost:10.37', 'Add Bonds: between 1 and 24 with Bond Type 2.0 and Cost -6.05', 'Remove Bonds: between 1 and 12, with Bond Type 1.0 and Cost -1.35', 'H number change -1 cost of atom 2: 0.18', 'H number change 1 cost of atom 13: 0.09'], ['Select Leaving Group with Index 6 and Cost 3.49', 'Initial Cost:7.66', 'Add Bonds: between 1 and 12 with Bond Type 2.0 and Cost -1.01', 'H number change -2 cost of atom 2: 1.16'], ['Select Leaving Group with Index 20 and Cost 2.84', 'Initial Cost:7.17', 'Replace Bonds: between 1 and 2, from Bond Type 2.0 to Bond Type1 with Cost -0.52', 'H number change 1 cost of atom 2: 2.22', 'H number change 1 cost of atom 3: 1.24'], ['Select Leaving Group with Index 3 and Cost 2.58', 'Initial Cost:11.32', 'Add Bonds: between 1 and 13 with Bond Type 1.0 and Cost -2.01', 'Remove Bonds: between 0 and 1, with Bond Type 1.0 and Cost -1.42', 'H number change 1 cost of atom 1: 1.12'], None, None, None]", 'route_cost': 30.16051983833313, 'route_len': 4} 
```

