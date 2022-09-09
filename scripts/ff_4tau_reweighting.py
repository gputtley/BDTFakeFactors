from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.ff_ml import ff_ml
from UserCode.BDTFakeFactors.functions import PrintDatasetSummary, MakeSelFromList, MakeSelDict
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

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='vvvloose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
parser.add_argument('--load_models_ff', help= 'Load model from file for fake factors',  action='store_true')
parser.add_argument('--load_models_correction', help= 'Load model from file for the corrections',  action='store_true')
parser.add_argument('--load_hyperparameters_ff', help= 'Load hyperparameters from file for fake factors',  action='store_true')
parser.add_argument('--load_hyperparameters_correction', help= 'Load hyperparameters from file for the corrections',  action='store_true')
parser.add_argument('--scan_ff', help= 'Do hyperparameter scan for fake factors',  action='store_true')
parser.add_argument('--scan_batch_ff', help= 'Do hyperparameter scan on the batch for fake factors',  action='store_true')
parser.add_argument('--scan_correction', help= 'Do hyperparameter scan for the corrections',  action='store_true')
parser.add_argument('--scan_batch_correction', help= 'Do hyperparameter scan on the batch for the corrections',  action='store_true')
parser.add_argument('--collect_scan_batch_ff', help= 'Collect hyperparameter scan on the batch for fake factors',  action='store_true')
parser.add_argument('--collect_scan_batch_correction', help= 'Collect hyperparameter scan on the batch for the corrections',  action='store_true')
parser.add_argument('--no_plots', help= 'Do not run plotting',  action='store_true')
parser.add_argument('--do_year_plots', help= 'Run split by year plotting',  action='store_true')
parser.add_argument('--do_train_test_plots', help= 'Run split by year plotting',  action='store_true')
parser.add_argument('--only_mc', help= 'Only run MC closure',  action='store_true')
args = parser.parse_args()

if args.batch:
  cmd = "python scripts/ff_4tau_reweighting.py"
  for a in vars(args):
    if "batch" in a: continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      cmd += " --{}={}".format(a,getattr(args, a))
  name = "ff_4tau_reweighting_job_{}_{}_{}".format(args.channel,args.pass_wp,args.fail_wp)
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  SubmitBatchJob("jobs/"+name+".sh")
  exit()


############ Inputs ##########################

years = ["2016_preVFP","2016_postVFP","2017","2018"]

var_file = open("variables/ff_4tau_{}.txt".format(args.channel),"r")
fitting_variables = var_file.read().split("\n")
var_file.close()
fitting_variables = [s.strip() for s in fitting_variables if s]

scoring_variables = fitting_variables + ["yr_2016_preVFP","yr_2016_postVFP","yr_2017","yr_2018"]

reweight_plot_variables = [
                           "pt_{}".format(args.channel.find("t")+1)
                           ]

closure_plot_variables = [
                          ["mvis_12",(60,0,300)],
#                          ["mvis_13",(60,0,300)],
#                          ["mvis_14",(60,0,300)],
#                          ["mvis_23",(60,0,300)],
#                          ["mvis_24",(60,0,300)],
#                          ["mvis_34",(60,0,300)],
#                          ["pt_1",(50,0,250)],
#                          ["pt_2",(50,0,250)],
#                          ["pt_3",(50,0,250)],
#                          ["pt_4",(50,0,250)],
                          ]

param_grid = {
              "n_estimators":[80,100,200,300],
              "max_depth":[2,3,4],
              "learning_rate":[0.06,0.08,0.1],
              "min_samples_leaf": [200,400,600,800],
              }

############ Functions ##########################

def ConvertDatasetName(pf_val,ch):
  return "dataframes/{}_{}_dataframe.pkl".format(ch,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("deepTauVsJets","dTvJ"))

def ConvertMCDatasetName(pf_val,proc,ch):
  return "dataframes/{}_{}_mc_{}_dataframe.pkl".format(ch,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("deepTauVsJets","dTvJ"),proc)

############# Assign dataframes #################

### setup sideband and signal regions ###

SIDEBAND = "(q_sum!=0)"
SIGNAL = "(q_sum==0)"

### setup pass and fail ###

if args.fail_wp == None:
  FAIL = "(deepTauVsJets_iso_X>=0 && deepTauVsJets_{}_X==0)".format(args.pass_wp)
else:
  FAIL = "(deepTauVsJets_{}_X && deepTauVsJets_{}_X==0)".format(args.fail_wp,args.pass_wp)

PASS = "(deepTauVsJets_{}_X==1)".format(args.pass_wp)


### Get all combination you need to loop through ###

lst_n = []
for ind,i in enumerate(args.channel):
  if i == "t":
    lst_n.append(ind+1)

total_keys = []
for ind,i in enumerate(lst_n):
  for k in list(itertools.combinations(lst_n, ind+1)):
    total_keys.append(k)

### Begin load and training loop ###

datasets_created = []
store = {}

for ind, t_ff in enumerate(total_keys):

  print "<< Running for {} >>".format(t_ff)

  t_ff_string = str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ","")
  sel = MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL)

  store[t_ff_string] = ff_ml()
  store[t_ff_string].AddDataframes(ConvertDatasetName(sel["Raw F_{F}"]["fail"],args.channel),ConvertDatasetName(sel["Raw F_{F}"]["pass"],args.channel),"Raw F_{F}")
  if ind + 1 != len(total_keys):
    store[t_ff_string].AddDataframes(ConvertDatasetName(sel["Alternative F_{F}"]["fail"],args.channel),ConvertDatasetName(sel["Alternative F_{F}"]["pass"],args.channel),"Alternative F_{F}")
    store[t_ff_string].AddDataframes(ConvertDatasetName(sel["Correction"]["fail"],args.channel),ConvertDatasetName(sel["Correction"]["pass"],args.channel),"Correction")
  else:
    for proc in ["W","DY","ttbar"]:
      store[t_ff_string].AddDataframes(ConvertMCDatasetName(sel["Raw F_{F}"]["fail"],proc,args.channel),ConvertMCDatasetName(sel["Raw F_{F}"]["pass"],proc,args.channel),"Alternative F_{F}")
      store[t_ff_string].AddDataframes(ConvertMCDatasetName(sel["Signal"]["fail"],proc,args.channel),ConvertMCDatasetName(sel["Signal"]["pass"],proc,args.channel),"Correction")

  for rwt_key in ["Raw F_{F}","Alternative F_{F}","Correction"]:

    if args.scan_batch_ff and "Correction" in rwt_key: continue

    rwt_key_name = rwt_key.lower().replace(" ","_").replace("_{","").replace("}","")

    train_fail, wt_train_fail, train_pass, wt_train_pass = store[t_ff_string].GetDataframes(rwt_key,data_type="train")
    test_fail, wt_test_fail, test_pass, wt_test_pass = store[t_ff_string].GetDataframes(rwt_key,data_type="test")

    model_name = "{}_{}_{}_{}_{}".format(args.channel,str(args.fail_wp),args.pass_wp,rwt_key.lower().replace(" ","_").replace("_{","").replace("}",""),str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ",""))

    if not ((args.load_models_ff and "F_{F}" in rwt_key) or (args.load_models_correction and "Correction" in rwt_key)):

      print "<< Running reweighting training for {} >>".format(rwt_key)

      rwter = reweighter.reweighter()

      if (args.scan_ff and "F_{F}" in rwt_key) or (args.scan_correction and "Correction" in rwt_key):

        rwter.grid_search(train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid, scoring_variables=scoring_variables)
        with open('hyperparameters/ff_hp_{}.json'.format(model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)

      elif (args.scan_batch_ff and "F_{F}" in rwt_key) or (args.scan_batch_correction and "Correction" in rwt_key):

        rwter.grid_search_batch(model_name,train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid, scoring_variables=scoring_variables)

      elif (args.collect_scan_batch_ff and "F_{F}" in rwt_key) or (args.collect_scan_batch_correction and "Correction" in rwt_key):

        rwter.collect_grid_search_batch(model_name)
        with open('hyperparameters/ff_hp_{}.json'.format(model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)
        rwter.norm_and_fit(train_fail, train_pass, wt_train_fail ,wt_train_pass)

      else:

        if (args.load_hyperparameters_ff and "F_{F}" in rwt_key) or (args.load_hyperparameters_correction and "Correction" in rwt_key):
          with open('hyperparameters/ff_hp_{}.json'.format(model_name)) as json_file: params = json.load(json_file)
          rwter.set_params(params)
        rwter.norm_and_fit(train_fail, train_pass, wt_train_fail ,wt_train_pass)

      if not (args.scan_batch_ff or args.scan_batch_correction): pkl.dump(rwter,open("BDTs/ff_{}.pkl".format(model_name), "wb"))

    else:

      print "<< Loading reweighting training for {} >>".format(rwt_key)

      rwter = pkl.load(open("BDTs/ff_{}.pkl".format(model_name), "rb"))

    store[t_ff_string].AddModel(rwter,rwt_key)


    ### Run plotting ###

    if not args.no_plots and not args.only_mc:

      # reweight plots

      for var in reweight_plot_variables:

        store[t_ff_string].PlotReweights(var, rwt_key, None, data_type=None, title_left=args.channel, title_right="all_years", plot_name="reweight_colz_plot_all_{}_{}_{}_{}_all_years".format(t_ff_string,var,rwt_key_name,args.channel))
        if args.do_train_test_plots:
          store[t_ff_string].PlotReweights(var, rwt_key, None, data_type="train", title_left=args.channel, title_right="all_years", plot_name="reweight_colz_plot_train_{}_{}_{}_{}_all_years".format(t_ff_string,var,rwt_key_name,args.channel))
          store[t_ff_string].PlotReweights(var, rwt_key, None, data_type="test", title_left=args.channel, title_right="all_years", plot_name="reweight_colz_plot_test_{}_{}_{}_{}_all_years".format(t_ff_string,var,rwt_key_name,args.channel))

        if args.do_year_plots:
          for year in years:
            store[t_ff_string].PlotReweights(var, rwt_key, "yr_{}==1".format(year), data_type=None, title_left=args.channel, title_right=year, plot_name="reweight_colz_plot_all_{}_{}_{}_{}".format(t_ff_string,var,rwt_key_name,args.channel,year))
            if args.do_train_test_plots:
              store[t_ff_string].PlotReweights(var, rwt_key, "yr_{}==1".format(year), data_type="train", title_left=args.channel, title_right=year, plot_name="reweight_colz_plot_train_{}_{}_{}_{}".format(t_ff_string,var,rwt_key_name,args.channel,year))
              store[t_ff_string].PlotReweights(var, rwt_key, "yr_{}==1".format(year), data_type="test", title_left=args.channel, title_right=year, plot_name="reweight_colz_plot_test_{}_{}_{}_{}".format(t_ff_string,var,rwt_key_name,args.channel,year))


      # closure plots

      for var in closure_plot_variables:

        store[t_ff_string].PlotClosure(var, rwt_key, None, data_type=None, title_left=args.channel, title_right="all_years", plot_name="closure_plot_all_{}_{}_{}_{}_all_years".format(t_ff_string,var[0],rwt_key_name,args.channel))
        if args.do_train_test_plots:
          store[t_ff_string].PlotClosure(var, rwt_key, None, data_type="train", title_left=args.channel, title_right="all_years", plot_name="closure_plot_train_{}_{}_{}_{}_all_years".format(t_ff_string,var[0],rwt_key_name,args.channel))
          store[t_ff_string].PlotClosure(var, rwt_key, None, data_type="test", title_left=args.channel, title_right="all_years", plot_name="closure_plot_test_{}_{}_{}_all_{}_years".format(t_ff_string,var[0],rwt_key_name,args.channel))

        if args.do_year_plots:
          for year in years:
            store[t_ff_string].PlotClosure(var, rwt_key, "yr_{}==1".format(year), data_type=None, title_left=args.channel, title_right=year, plot_name="closure_plot_all_{}_{}_{}_{}_{}".format(t_ff_string,var[0],rwt_key_name,args.channel,year))
            if args.do_train_test_plots:
              store[t_ff_string].PlotClosure(var, rwt_key, "yr_{}==1".format(year), data_type="train", title_left=args.channel, title_right=year, plot_name="closure_plot_train_{}_{}_{}_{}_{}".format(t_ff_string,var[0],rwt_key_name,args.channel,year))
              store[t_ff_string].PlotClosure(var, rwt_key, "yr_{}==1".format(year), data_type="test", title_left=args.channel, title_right=year, plot_name="closure_plot_test_{}_{}_{}_{}_{}".format(t_ff_string,var[0],rwt_key_name,args.channel,year))


# run full signal region mc closures

if not args.no_plots:

  proc_colours = {"DY":8,"W":46,"ttbar":32}
  
  for proc in ["W","DY","ttbar"]:
    for ind, t_ff in enumerate(total_keys):
  
      t_ff_string = str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ","")
      sel = MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL)
      store[t_ff_string].AddDataframes(ConvertMCDatasetName(sel["Signal"]["fail"],proc,args.channel),ConvertMCDatasetName(sel["Signal"]["pass"],proc,args.channel),"Signal")
      temp_df1, _ = store[t_ff_string].GetDataframes("Signal",full=True,with_reweights=True)
  
      if ((len(t_ff) % 2) == 0):
        temp_df1.loc[:,"weights"] *= -1.0
  
      if ind == 0:
        df1 = temp_df1.copy(deep=True)
      else:
        df1 = pd.concat([df1,temp_df1.copy(deep=True)], ignore_index=True, sort=False)
  
    new_filename = "dataframes/{}_{}_combined_signal_region_dataframe.pkl".format(args.channel,proc)
    df1.to_pickle(new_filename) 
    del df1
    store[t_ff_string].df1["Signal"] = [new_filename]
  

    for var in closure_plot_variables:
  
      store[t_ff_string].PlotClosure(var, "Signal", None, data_type=None, title_left=args.channel, title_right="all_years", plot_name="closure_plot_all_{}_mc_{}_{}_{}_all_years".format(proc,var[0],rwt_key_name,args.channel), fillcolour=proc_colours[proc])
      if args.do_year_plots:
        for year in years:
          store[t_ff_string].PlotClosure(var, "Signal", "yr_{}==1".format(year), data_type=None, title_left=args.channel, title_right=year, plot_name="closure_plot_all_{}_mc_{}_{}_{}_{}".format(proc,var[0],rwt_key_name,args.channel,year), fillcolour=proc_colours[proc])

    # reset dataframe list
    for ind, t_ff in enumerate(total_keys):
      t_ff_string = str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ","")
      store[t_ff_string].df1["Signal"] = []
      store[t_ff_string].df2["Signal"] = []

      

