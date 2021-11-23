> Authors:
> * Nikolas Borrel (<nibor@elektro.dtu.dk>)
> * Allan P. Engsig-Karup (<apek@dtu.dk>)
> * Cheol-Ho Jeong (<chj@elektro.dtu.dk>)

# About
Code for sound field predictions in domains with Neumann and impedance boundaries. Used for generating results from the paper "Physics-informed neural networks for 1D sound field predictions with parameterized sources and impedance boundaries" by N. Borrel-Jensen, A. P. Engsig-Karup, and C. Jeong.

# Run
## Train
Run
```bash
python3 main_train.py --path_settings="path/to/script.json"
```
Scripts for setting up models with Neumann, frequency-independent and dependent boundaries can be found in scripts/settings (see `JSON settings`).

## Evaluate
Run
```bash
python3 main_evaluate.py
```
The settings are
```python
id_dir = <unique id>
settings_filename = 'settings.json'
base_dir = "path/to/base/dir"

do_plots_for_paper = <bool>
do_animations = <bool>
do_side_by_side_plot = <bool>
```
The `id_dir` corresponds to the output directory generated after training, `settings_filename` is the name of the settings file used for training (located inside the `id_dir` directory), `base_dir` is the path to the base directory (see `Input/output directory structure`).

## Evaluate model execution time
To evaluate the execution time of the surrogate model, run
```bash
python3 main_evaluate_timings.py --path_settings="path/to/script.json" --trained_model_tag="trained-model-dir"
```
The `trained_model_tag` is the directory with the trained model weights trained using the scripts located at the path given in `path_settings`.

# Setup

## Dependencies
The library dependencies can be installed with pip and are
* tensorflow==2.5.1 
* sciann==0.6.4.7 
* matplotlib 
* pandas 
* smt 
* pydot 
* graphviz

See also `scripts/package_install.sh`.

## Input/output directory structure
The input data should be located inside a folder `base_dir` (can be given any name) with a specific relative directory structure as 

```verbatim
base_dir/    
    reference_data/
        freq_dep_1D_1000.00Hz_sigma0.2_c1_d0.02_srcs3.hdf5
        ...
        freq_indep_1D_1000.00Hz_sigma0.2_c1_xi5.83_srcs3.hdf5
        ...
        neumann_1D_1000.00Hz_sigma0.2_c1_srcs3.hdf5
        ...
    ...
```
The reference data from the paper can be downloaded [here](http://www.todo.com). The reference data is generated using an SEM solver for impedance boundaries, whereas the Python script `main_generate_analytical_data.py` was used for Neumann boundaries.

Note: the folder `reference_data` should be named exactly as stated.

Output result data will be located inside the `results` folder as
```verbatim
base_dir/
    ...
    results/
        id_folder/
            figs/
            models/
                LossType.PINN/
                    checkpoint
                    cp.ckpt.data-00000-of-00001
                    cp.ckpt.index
            settings.json
```
 The `settings.json` file is identical to the settings file used for training indicated by the ``--path_settings`` argument. The directory `LossType.PINN` contains the trained model weights.

The results from the paper can be downloaded here [here](http://www.todo.com) and should be placed inside the `results` directory (run `python3 main_evaluate.py` to evaluate).

For transfer learning, the model to continue the training on, should be located
```verbatim
base_path/
    trained_models/
        trained_model_tag/
            checkpoint
            cp.ckpt.data-00000-of-00001
            cp.ckpt.index
    ...
```
where the directory `trained_model_tag/` would correspond to the generated `LossType.PINN/` model created after training (can be given any name/tag).

## JSON settings
The script `scripts/settings/neumann.json` was used for training the Neumann model from the paper
```json
{
    "id": "neumann_srcs3_sine_3_256_7sources_loss02",
    "base_path": "path/to/base_dir/",
    
    "c": 1,
    "c_phys": 343,
    "___COMMENT_fmax___": "1000Hz*c/343 = 2.9155 for c=1",
    "fmax": 2.9155,

    "tmax": 4,
    "Xmin": [-1],
    "Xmax": [1],
    "source_pos": [[-0.3],[-0.2],[-0.1],[0.0],[0.1],[0.2],[0.3]],
    
    "sigma0": 0.2,
    "rho": 1.2,
    "ppw": 10,

    "epochs": 25000,
    "stop_loss_value": 0.0002,
    
    "boundary_type": "NEUMANN",
    "data_filename": "neumann_1D_1000.00Hz_sigma0.2_c1_srcs7.hdf5",
    
    "batch_size": 512,
    "learning_rate": 0.0001,
    "optimizer": "adam",

    "__comment0__": "NN setting for the PDE",
    "activation": "sin",
    "num_layers": 3,
    "num_neurons": 256,

    "ic_points_distr": 0.25,
    "bc_points_distr": 0.45,

    "loss_penalties": {
        "pde":1,
        "ic":20,
        "bc":1
    },

    "verbose_out": false,
    "show_plots": false
}
```
The script `scripts/settings/freq_indep.json` was used for training the frequency-independent model from the paper
```json
{
    "id": "freq_indep_sine_3_256_7sources_loss02",
    "base_path": "path/to/base_dir/",

    "c": 1,
    "c_phys": 343,
    "___COMMENT_fmax___": "1000Hz*c/343 = 2.9155 for c=1",
    "fmax": 2.9155,

    "tmax": 4,
    "Xmin": [-1],
    "Xmax": [1],
    "source_pos": [[-0.3],[-0.2],[-0.1],[0.0],[0.1],[0.2],[0.3]],
    
    "sigma0": 0.2,
    "rho": 1.2,
    "ppw": 10,

    "epochs": 25000,
    "stop_loss_value": 0.0002,
    
    "batch_size": 512,
    "learning_rate": 0.0001,
    "optimizer": "adam",

    "boundary_type": "IMPEDANCE_FREQ_INDEP",
    "data_filename": "freq_indep_1D_1000.00Hz_sigma0.2_c1_xi5.83_srcs7.hdf5",

    "__comment0__": "NN setting for the PDE",
    "activation": "sin",
    "num_layers": 3,
    "num_neurons": 256,

    "impedance_data": {
        "__comment1__": "xi is the acoustic impedance ONLY for freq. indep. boundaries",
        "xi": 5.83
    },

    "ic_points_distr": 0.25,
    "bc_points_distr": 0.45,
    
    "loss_penalties": {
        "pde":1,
        "ic":20,
        "bc":1
    },

    "verbose_out": false,
    "show_plots": false
}
```
The script `scripts/settings/freq_dep.json` was used for training the frequency-dependent model from the paper
```json
{
    "id": "freq_dep_sine_3_256_7sources_d01",
    "base_path": "path/to/base_dir/",

    "c": 1,
    "c_phys": 343,
    "___COMMENT_fmax___": "1000Hz*c/343 = 2.9155 for c=1",
    "fmax": 2.9155,

    "tmax": 4,
    "Xmin": [-1],
    "Xmax": [1],
    "source_pos": [[-0.3],[-0.2],[-0.1],[0.0],[0.1],[0.2],[0.3]],
    
    "sigma0": 0.2,
    "rho": 1.2,
    "ppw": 10,

    "epochs": 50000,
    "stop_loss_value": 0.0002,

    "do_transfer_learning": false,

    "boundary_type": "IMPEDANCE_FREQ_DEP",
    "data_filename": "freq_dep_1D_1000.00Hz_sigma0.2_c1_d0.10_srcs7.hdf5",
    
    "batch_size": 512,
    "learning_rate": 0.0001,
    "optimizer": "adam",

    "__comment0__": "NN setting for the PDE",
    "activation": "sin",
    "num_layers": 3,
    "num_neurons": 256,

    "__comment1__": "NN setting for the auxillary differential ODE",
    "activation_ade": "tanh",
    "num_layers_ade": 3,
    "num_neurons_ade": 20,

    "impedance_data": {
        "d": 0.1,
        "type": "IMPEDANCE_FREQ_DEP",
        "lambdas": [7.1109025021758407,205.64002739443146],
        "alpha": [6.1969460587749818],
        "beta": [-15.797795759219973],
        "Yinf": 0.76935257750377573,
        "A": [-7.7594660571346719,0.0096108036858666163],
        "B": [-0.016951521199665469],
        "C": [-2.4690553703530442]
      },

    "accumulator_factors": [10.26, 261.37, 45.88, 21.99],

    "ic_points_distr": 0.25,
    "bc_points_distr": 0.45,

    "loss_penalties": {
        "pde":1,
        "ic":20,
        "bc":1,
        "ade":[10,10,10,10]
    },

    "verbose_out": false,
    "show_plots": false
}
```
The `base_path` should be adjusted to your local setup as described in section "Input/output directory structure".

## HPC (DTU)
The scripts for training the models on the GPULAB clusters at DTU are located at `scripts/settings/run_*.sh`.

## VSCode
Launch scripts for VS Code are located inside `.vscode` and running the settings script `local_train.json` in debug mode is done selecting the `Python: TRAIN` scheme (open `pinn-acoustics.code-workspace` to enable the workspace).

# License
See [LICENSE](LICENSE)