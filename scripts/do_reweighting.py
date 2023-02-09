from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.ff_ml import ff_ml
from UserCode.BDTFakeFactors.functions import PrintDatasetSummary
from UserCode.BDTFakeFactors.batch import CreateBatchJob,SubmitBatchJob
from UserCode.BDTFakeFactors import reweighter
from collections import OrderedDict
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import itertools
import os
import json
import copy
import gc
import glob
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',help= 'config to input', default='mmtt.yaml')
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
parser.add_argument('--load_models', help= 'Load model from file for fake factors',  action='store_true')
parser.add_argument('--load_hyperparameters', help= 'Load hyperparameters from file for fake factors',  action='store_true')
parser.add_argument('--scan', help= 'Do hyperparameter scan for fake factors',  action='store_true')
parser.add_argument('--scan_batch', help= 'Do hyperparameter scan on the batch for fake factors',  action='store_true')
parser.add_argument('--collect_scan_batch', help= 'Collect hyperparameter scan on the batch for fake factors',  action='store_true')
parser.add_argument('--no_plots', help= 'Do not run plotting',  action='store_true')
parser.add_argument('--do_year_plots', help= 'Run split by year plotting',  action='store_true')
parser.add_argument('--do_train_test_plots', help= 'Run split by year plotting',  action='store_true')
args = parser.parse_args()

############ Import config ####################

with open(args.cfg) as f: data = yaml.load(f)

############ Batch Submission ####################

if args.batch:
  cmd = "python scripts/do_reweighting.py"
  for a in vars(args):
    if a == "batch": continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      cmd += " --{}={}".format(a,getattr(args, a))
  name = "reweighting_job_{}".format(data["channel"])
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  if data["channel"] == "ttt":
    SubmitBatchJob("jobs/"+name+".sh",time=500)
  else:
    SubmitBatchJob("jobs/"+name+".sh")
  exit()

if args.scan_batch:
  args.no_plots = True

############# Parameter grid for scan  #################

param_grid = {
              "n_estimators":[400,600,800,1000,1200],
              "max_depth":[2,3,4],
              "learning_rate":[0.03,0.04,0.04],
              "min_samples_leaf": [400,800,1200],
              "loss_regularization": [5,6]
              }

#param_grid = {
#              "n_estimators":[1200,1400],
#              "max_depth":[2,3],
#              "learning_rate":[0.03,0.04],
#              "min_samples_leaf": [1200],
#              "loss_regularization": [6]
#              }

############ Functions ##########################

############# Assign dataframes #################

### Begin load and training loop ###

if not os.path.isdir("BDTs/{}".format(data["channel"])): os.system("mkdir BDTs/{}".format(data["channel"]))
if not os.path.isdir("hyperparameters/{}".format(data["channel"])): os.system("mkdir hyperparameters/{}".format(data["channel"]))

datasets_created = []
store = {}

  
store = ff_ml()
for f in glob.glob("dataframes/{}/subtracted_fail/subtracted_fail_C*".format(data["channel"])): store.AddDataframes(f,f.replace("fail","pass"),"C")
for f in glob.glob("dataframes/{}/subtracted_fail/subtracted_fail_D*".format(data["channel"])): store.AddDataframes(f,f.replace("fail","pass"),"D")
for f in glob.glob("dataframes/{}/subtracted_fail/subtracted_fail_B*".format(data["channel"])): store.AddDataframes(f,f.replace("fail","pass"),"B")

model_name = "reweights_{}".format(data["channel"])

if not args.load_models:

  rwter = reweighter.reweighter(n_estimators=600, learning_rate=0.03, min_samples_leaf=1200, max_depth=2, loss_regularization=5)

  if not args.collect_scan_batch:
    train_fail, wt_train_fail, train_pass, wt_train_pass = store.GetDataframes("All",data_type="train")
    test_fail, wt_test_fail, test_pass, wt_test_pass = store.GetDataframes("All",data_type="test")

  if args.scan:
 
    print "<< Running grid search reweighting training >>"
    rwter.grid_search(train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid,)
    with open('hyperparameters/{}/ff_hp_{}.json'.format(data["channel"],model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)

  elif args.scan_batch:

    print "<< Batch submitting grid search reweighting training >>"
    rwter.grid_search_batch(model_name,train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid)

  elif args.collect_scan_batch:

    print "<< Collecting reweighting training for >>"
    rwter = rwter.collect_grid_search_batch(model_name)
    with open('hyperparameters/{}/ff_hp_{}.json'.format(data["channel"],model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)

  else:

    if args.load_hyperparameters:
      with open('hyperparameters/{}/ff_hp_{}.json'.format(data["channel"],model_name)) as json_file: params = json.load(json_file)
      rwter.set_params(params)

    print "<< Running reweighting training for >>"
    rwter.norm_and_fit(train_fail, train_pass, wt_train_fail ,wt_train_pass)

  if not args.scan_batch: pkl.dump(rwter,open("BDTs/{}/ff_{}.pkl".format(data["channel"],model_name), "wb"))

else:

  print "<< Loading reweighting training for >>"
  rwter = pkl.load(open("BDTs/{}/ff_{}.pkl".format(data["channel"],model_name), "rb"))

### Run plotting ###
if not args.no_plots:

  print "<< Running plotting >>"

  store.AddModel(rwter,"All")
  tp = [None]
  if args.do_train_test_plots: tp += ["train","test"]

  yp = ["all_years"]
  if args.do_year_plots: yp += years

  for tt in tp:
    for yr in yp:

      if yr != "all_years":
        yr_sel = "yr_{}==1".format(yr)
      else:
        yr_sel = None

      if tt == None:
        tt_name = "all"
      else:
        tt_name = tt

      store.PlotReweightsAndClosure("All", yr_sel, data_type=tt, title_left=data["channel"], title_right=yr, plot_name="{}_{}_{}".format(tt_name,data["channel"],yr))
