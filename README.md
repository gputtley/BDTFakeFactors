# BDTFakeFactors

This directory is to produce fake factors for hadronic taus using the BDT reweighting method described here https://arogozhnikov.github.io/hep_ml/reweight.html.

## Json Selection

All of the files and selection used in training are stored in the json_selection folder. Currently this is set up for the 4tau only.

## Variables used
.txt files of the variables used in the training are stored in the variables folder.

## Making Dataframes

The dataframes are made from the json_selection files and stored in the dataframes folder by running the following command.

```bash
python scripts/ff_4tau_dataframes.py --channel=mmtt --batch
```

## Running reweights - simple

To run the reweights simply you can run the following command. This will produce reweight and closure plots and put these in the plots folder as well as saving the BDTs and hyperparameters in the relevant folder.

```bash
python scripts/ff_4tau_reweighting.py --channel=mmtt --batch
```

## Running reweights - hyperparameter scans

To find the best model, you will need to run hyperparameter scans on both the fake factors and corrections. You can do this with the following commands.

```bash
python scripts/ff_4tau_reweighting.py --channel=mmtt --scan_batch_ff --no_plots
python scripts/ff_4tau_reweighting.py --channel=mmtt --collect_scan_batch_ff  --no_plots --batch
python scripts/ff_4tau_reweighting.py --channel=mmtt --load_models_ff --scan_batch_correction --no_plots
python scripts/ff_4tau_reweighting.py --channel=mmtt --load_models_ff --collect_scan_batch_correction --batch
```
