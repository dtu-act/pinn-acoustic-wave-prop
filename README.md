> Authors:
> * Nikolas Borrel (<nibor@elektro.dtu.dk>)
> * Allan P. Engsig-Karup (<apek@dtu.dk>)
> * Cheol-Ho Jeong (<chj@elektro.dtu.dk>)

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

# Settings

## Input/output directory structure
The input data should be located in a specific relative directory structure as (data used for the paper can be downloaded [here](http://www.todo.com))

```verbatim
base_path/
    trained_models/
        trained_model_tag/
            checkpoint
            cp.ckpt.data-00000-of-00001
            cp.ckpt.index
    training_data/
        freq_dep_1D_2000.00Hz_sigma0.2_c1_d0.02_srcs3.hdf5
        ...
        freq_indep_1D_2000.00Hz_sigma0.2_c1_xi5.83_srcs3.hdf5
        ...
        neumann_1D_2000.00Hz_sigma0.2_c1_srcs3.hdf5
        ...
```
The reference data are located inside the `training_data/` directory generated, where the data for impedance boundaries are generated using our SEM simulator, and for Neumann boundaries, the Python script `main_generate_analytical_data.py` was used.

Output result data are located inside the `results` folder
```verbatim
base_path/
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

## JSON settings
The script `scripts/settings/neumann.json` was used for training the Neumann model from the paper
```json
{
    "id": "neumann_srcs3_sine_3_256_7sources_loss02",
    "base_dir": "../data/pinn",
    
    "c": 1,
    "c_phys": 343,
    "___COMMENT_fmax___": "2000Hz*c/343 = 5.8309 for c=1, =23.3236 for c=4",
    "fmax": 5.8309,

    "tmax": 4,
    "xmin": -1,
    "xmax": 1,
    "source_pos": [-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3],
    
    "sigma0": 0.2,
    "rho": 1.2,
    "ppw": 5,

    "epochs": 25000,
    "stop_loss_value": 0.0002,
    
    "boundary_type": "NEUMANN",
    "data_filename": "neumann_1D_2000.00Hz_sigma0.2_c1_srcs7.hdf5",
    
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
The script `scripts/settings/freq_indep.json` was used for training the Neumann model from the paper
```json
{
    "id": "freq_indep_sine_3_256_7sources_loss02",
    "base_dir": "../data/pinn",

    "c": 1,
    "c_phys": 343,
    "___COMMENT_fmax___": "2000Hz*c/343 = 5.8309 for c=1, =23.3236 for c=4",
    "fmax": 5.8309,

    "tmax": 4,
    "xmin": -1,
    "xmax": 1,
    "source_pos": [-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3],
    
    "sigma0": 0.2,
    "rho": 1.2,
    "ppw": 5,

    "epochs": 25000,
    "stop_loss_value": 0.0002,
    
    "batch_size": 512,
    "learning_rate": 0.0001,
    "optimizer": "adam",

    "boundary_type": "IMPEDANCE_FREQ_INDEP",
    "data_filename": "freq_indep_1D_2000.00Hz_sigma0.2_c1_xi5.83_srcs7.hdf5",

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
The script `scripts/settings/freq_dep.json` was used for training the Neumann model from the paper
```json
{
    "id": "freq_dep_sine_3_256_7sources_d01",
    "base_dir": "../data/pinn",

    "c": 1,
    "c_phys": 343,
    "___COMMENT_fmax___": "2000Hz*c/343 = 5.8309 for c=1, =23.3236 for c=4",
    "fmax": 5.8309,

    "tmax": 4,
    "xmin": -1,
    "xmax": 1,
    "source_pos": [-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3],
    
    "sigma0": 0.2,
    "rho": 1.2,
    "ppw": 5,

    "epochs": 50000,
    "stop_loss_value": 0.0002,

    "do_transfer_learning": false,

    "boundary_type": "IMPEDANCE_FREQ_DEP",
    "data_filename": "freq_dep_1D_2000.00Hz_sigma0.2_c1_d0.10_srcs7.hdf5",
    
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

## HPC (DTU)
The scripts for training the models on the GPULAB clusters at DTU are located at `scripts/settings/run_*.sh`.

## VSCode
Launch scripts for VS Code are located inside `.vscode` and running the settings script `local_train.json` in debug mode is done selecting the `Python: TRAIN` scheme (open `pinn-acoustics.code-workspace` to enable the workspace).

# License
See [LICENSE](LICENSE)