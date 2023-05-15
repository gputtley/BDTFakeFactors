from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.ff_ml import ff_ml
from UserCode.BDTFakeFactors.functions import PrintDatasetSummary,EqualBinStats
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
parser.add_argument('--add_uncerts',help= 'Json file of uncertainties to add to plots', default=None)
parser.add_argument('--specific_tau',help= 'Run for specific tau in the event', default="")
parser.add_argument('--shift',help= 'Shift MC yields up and down by 10%', default='', choices=["","up","down"])
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
  name = "reweighting_job{}_{}".format(args.shift,data["channel"])
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  if data["channel"] == "ttt":
    SubmitBatchJob("jobs/"+name+".sh")
  else:
    SubmitBatchJob("jobs/"+name+".sh")
  exit()

if args.scan_batch:
  args.no_plots = True

############# Parameter grid for scan  #################

#{u'n_estimators': 800, u'learning_rate': 0.02, u'min_samples_leaf': 5000, u'max_depth': 1, u'loss_regularization': 6}

param_grid = {
              "n_estimators":[80,100,200,300,400,600,800,1000,1200],
              "max_depth":[2,3],
              "learning_rate":[0.005,0.01,0.02,0.04,0.06],
              "min_samples_leaf": [1000,2000,3000],
              "loss_regularization": [6]
              }


#param_grid = {
#              "n_estimators":[600,800,1000],
#              "max_depth":[2,3],
#              "learning_rate":[0.04,0.05,0.06,0.07,0.08],
#              "min_samples_leaf": [600,800,1000,1200,1400,1600],
#              "loss_regularization": [4,5,6,7,8]
#              }
#
#param_grid = {
#              "n_estimators":[100,200],
#              "max_depth":[1,2],
#              "learning_rate":[0.005,0.01,0.02],
#              "min_samples_leaf": [3000,4000,5000],
#              "loss_regularization": [6]
#              }
#
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

if args.shift != "": args.shift = "_"+args.shift
if args.specific_tau != "": args.specific_tau = "_"+args.specific_tau
for f in glob.glob("dataframes/{}/subtracted_fail{}/subtracted_fail_C{}*".format(data["channel"],args.shift,args.specific_tau)): store.AddDataframes(f,f.replace("fail","pass"),"C")
for f in glob.glob("dataframes/{}/subtracted_fail{}/subtracted_fail_D{}*".format(data["channel"],args.shift,args.specific_tau)): store.AddDataframes(f,f.replace("fail","pass"),"D")
for f in glob.glob("dataframes/{}/subtracted_fail{}/subtracted_fail_B{}*".format(data["channel"],args.shift,args.specific_tau)): store.AddDataframes(f,f.replace("fail","pass"),"B")

model_name = "reweights_{}".format(data["channel"])

if not args.load_models:

  if data["channel"] == "ttt":
    rwter = reweighter.reweighter(n_estimators=400, learning_rate=0.04, min_samples_leaf=800, max_depth=3, loss_regularization=6)
  else:
    rwter = reweighter.reweighter(n_estimators=600, learning_rate=0.03, min_samples_leaf=1200, max_depth=2, loss_regularization=5)


  if not args.collect_scan_batch:
    train_fail, wt_train_fail, train_pass, wt_train_pass = store.GetDataframes("All",data_type="train")
    test_fail, wt_test_fail, test_pass, wt_test_pass = store.GetDataframes("All",data_type="test")


    if "normalize_in" in data:
      train_fail.loc[:,"weights"] = wt_train_fail
      train_pass.loc[:,"weights"] = wt_train_pass
      test_fail.loc[:,"weights"] = wt_test_fail
      test_pass.loc[:,"weights"] = wt_test_pass

      feature_names = []
      feature_sels = []
      for norm_feature in data["normalize_in"]:
        if type(norm_feature) != dict:
          feature_names.append(norm_feature)
          feature_sels.append(["(X.loc[:,\"%(norm_feature)s\"] == %(c)s)" % vars() for c in train_pass.loc[:,norm_feature].unique()])
        else:
          norm_feature_name = norm_feature.keys()[0]
          feature_names.append(norm_feature_name)
          feature_sels.append([])
          for cat in norm_feature[norm_feature_name]:
            tmp_feature_sel = "(" 
            for ind_val, val in enumerate(cat):
              if ind_val != 0: tmp_feature_sel += " | "
              tmp_feature_sel += "(X.loc[:,\"%(norm_feature_name)s\"] == %(val)s)" % vars()
            tmp_feature_sel += ")"
            feature_sels[-1].append(tmp_feature_sel)
 
      cut_list = [" & ".join(i) for i in list(itertools.product(*feature_sels))]
      total_scale = train_pass.loc[:,"weights"].sum()/float(len(cut_list))
      print "Total Scale", total_scale
      for cut in cut_list:
        wt = train_pass.loc[eval(cut.replace("X","train_pass")),"weights"].sum()
        thresh = 50
        scale_max = 2.0
        print "fail:", train_fail.loc[eval(cut.replace("X","train_fail")),"weights"].sum(), "pass:",train_pass.loc[eval(cut.replace("X","train_pass")),"weights"].sum()
        if wt == 0: continue
        scale = total_scale/wt
        if wt < thresh and scale > scale_max:
        #if scale > scale_max: 
          print "Stats in bin too low, capping normalising"
          scale = scale_max*1.0
        print cut, scale
        train_fail.loc[eval(cut.replace("X","train_fail")),"weights"] *= scale
        train_pass.loc[eval(cut.replace("X","train_pass")),"weights"] *= scale
        test_fail.loc[eval(cut.replace("X","test_fail")),"weights"] *= scale
        test_pass.loc[eval(cut.replace("X","test_pass")),"weights"] *= scale
        #print train_fail.loc[eval(cut.replace("X","train_fail")),"weights"].sum(), train_pass.loc[eval(cut.replace("X","train_pass")),"weights"].sum(),train_fail.loc[eval(cut.replace("X","train_fail")),"weights"].sum()/train_pass.loc[eval(cut.replace("X","train_pass")),"weights"].sum(), scale


      wt_train_fail = train_fail.loc[:,"weights"]
      wt_train_pass = train_pass.loc[:,"weights"] 

      train_fail = train_fail.drop("weights",axis=1)
      train_pass = train_pass.drop("weights",axis=1)

      wt_test_fail = test_fail.loc[:,"weights"]
      wt_test_pass = test_pass.loc[:,"weights"]

      test_fail = test_fail.drop("weights",axis=1)
      test_pass = test_pass.drop("weights",axis=1)



  if args.scan:
 
    print "<< Running grid search reweighting training >>"
    rwter.grid_search(train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid, scoring_variables=list(train_fail.columns))
    with open('hyperparameters/{}/ff_hp_{}.json'.format(data["channel"],model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)

  elif args.scan_batch:

    print "<< Batch submitting grid search reweighting training >>"
    rwter.grid_search_batch(model_name,train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid, scoring_variables=list(train_fail.columns))

  elif args.collect_scan_batch:

    print "<< Collecting reweighting training for >>"
    rwter = rwter.collect_grid_search_batch(model_name)
    with open('hyperparameters/{}/ff_hp_{}.json'.format(data["channel"],model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)

  else:

    if args.load_hyperparameters:
      hpf = 'hyperparameters/{}/ff_hp_{}.json'.format(data["channel"],model_name)
      if os.path.isfile(hpf): 
        with open(hpf) as json_file: params = json.load(json_file)
        rwter.set_params(params)

    print "<< Running reweighting training for >>"

    if "equal_stat_bins" in data:
      for ind,_ in enumerate(data["equal_stat_bins"]):
        rwter.wt_function.append(EqualBinStats(train_fail,
                                               wt_train_fail,data["equal_stat_bins"][ind]["col"],
                                               ignore_quantile=data["equal_stat_bins"][ind]["ignore_quantile"],
                                               n_bins=data["equal_stat_bins"][ind]["n_bins"],
                                               min_bin=data["equal_stat_bins"][ind]["min_bin"],
                                               max_bin=data["equal_stat_bins"][ind]["max_bin"]))
        rwter.wt_function_var.append(data["equal_stat_bins"][ind]["col"])

    rwter.norm_and_fit(train_fail, train_pass, wt_train_fail ,wt_train_pass)

    

  if not args.scan_batch: pkl.dump(rwter,open("BDTs/{}/ff_{}{}.pkl".format(data["channel"],model_name,args.shift), "wb"))

else:

  print "<< Loading reweighting training for >>"
  rwter = pkl.load(open("BDTs/{}/ff_{}{}.pkl".format(data["channel"],model_name,args.shift), "rb"))


### Run plotting ###
if not args.no_plots:

  print "<< Running plotting >>"

  store.AddModel(rwter,"All")
  tp = [None]
  if args.do_train_test_plots: tp += ["train","test"]

  yp = ["all_years"]
  if args.do_year_plots: yp += data["years_and_lumi"].keys()

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

      store.PlotReweightsAndClosure("All", yr_sel, data_type=tt, title_left=data["channel"], title_right=yr, plot_name="{}_{}_{}{}".format(tt_name,data["channel"],yr,args.shift), add_uncerts=args.add_uncerts)
