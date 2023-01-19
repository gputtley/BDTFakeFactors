# BDTFakeFactors

This directory is to produce fake factors for hadronic taus using the BDT reweighting method described here https://arogozhnikov.github.io/hep_ml/reweight.html.

## Inputs

The input folder is used to tell the algorithm what specific root files, selections and variables need to be used. The first layer of folders is the analysis name, the second layer of folders are the channels used for this specific file. The algorithm will fit fake factors for whenever "t" appears in the channel name. These both need to be parsed as an option to the dataframe and reweighting code.

Within these folders are a selection and variables folder, as well as a config json file. The selection folder contains information about file location, file names and selections needed to make the inputs for training. The variables folder requires three txt files: the fitting variables, the scoring variables (for hyperparameter scans) and plotting variables with their binning. The remaining config json sets the specific information about what type of fake factors you want to divide.

## Making Dataframes

The dataframes are made from the json_selection files and stored in the dataframes folder by running the following command.

```bash
python scripts/ff_dataframes.py --channel=mmtt --batch
```

## Running reweights - simple

To run the reweights simply you can run the following command. This will produce reweight and closure plots and put these in the plots folder as well as saving the BDTs and hyperparameters in the relevant folder.

```bash
python scripts/ff_reweighting.py --channel=mmtt --analysis=4tau --batch
```

## Running reweights - hyperparameter scans

To find the best model, you will need to run hyperparameter scans on both the fake factors and corrections. You can do this with the following commands.

```bash
python scripts/ff_reweighting.py --channel=mmtt --analysis=4tau --scan_batch_ff --no_plots
python scripts/ff_reweighting.py --channel=mmtt --analysis=4tau --collect_scan_batch_ff  --no_correction --no_plots --batch
python scripts/ff_reweighting.py --channel=mmtt --analysis=4tau --load_models_ff --scan_batch_correction --no_plots
python scripts/ff_reweighting.py --channel=mmtt --analysis=4tau --load_models_ff --collect_scan_batch_correction --batch
```
