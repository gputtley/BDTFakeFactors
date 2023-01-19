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
import copy
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--analysis',help= 'Analysis to train BDT for', default='4tau')
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
parser.add_argument('--no_correction', help= 'Do not run correction',  action='store_true')
args = parser.parse_args()

if args.batch:
  cmd = "python scripts/ff_reweighting_new.py"
  for a in vars(args):
    if a == "batch": continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      cmd += " --{}={}".format(a,getattr(args, a))
  name = "ff_reweighting_job_{}_{}_{}".format(args.channel,args.pass_wp,args.fail_wp)
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  SubmitBatchJob("jobs/"+name+".sh",time=300)
  exit()

if args.scan_batch_ff or args.scan_batch_correction:
  args.no_plots = True

############ Import config ####################

with open("input/{}/{}/config.json".format(args.analysis,args.channel)) as jsonfile:
  config = json.load(jsonfile)

years = config["years"]
SIDEBAND = config["DR"]
SIGNAL = config["SR"]
mc_procs = config["mc_procs"]
multiple_ff = config["multiple_ff"]
only_do_ff = config["only_do_ff"]
do_subtraction = config["do_subtraction"]
combine_taus = config["combine_taus"]

############# Variables needed #################

var_fitting_file = open("input/{}/{}/variables/fitting.txt".format(args.analysis,args.channel),"r")
fitting_variables = var_fitting_file.read().split("\n")
var_fitting_file.close()
fitting_variables = [s.strip() for s in fitting_variables if s]  + ["yr_"+y for y in years]

var_scoring_file = open("input/{}/{}/variables/scoring.txt".format(args.analysis,args.channel),"r")
scoring_variables = var_scoring_file.read().split("\n")
var_scoring_file.close()
scoring_variables = [s.strip() for s in scoring_variables if s] + ["yr_"+y for y in years]

var_plotting_file = open("input/{}/{}/variables/plotting.txt".format(args.analysis,args.channel),"r")
plotting_variables = var_plotting_file.read().split("\n")
var_plotting_file.close()
plotting_variables = [s.strip() for s in plotting_variables if s]

closure_plot_variables = []
closure_plot_binnings = []
for var in plotting_variables:
  if "[" in var:
    var_name = var.split("[")[0]
    bins = var.split("[")[1].split("]")[0].split(",")
    bins_float = [float(i) for i in bins]
  elif "(" in var:
    var_name = var.split("(")[0]
    bins = var.split("(")[1].split(")")[0].split(",")
    bins_float = tuple([int(bins[0]),float(bins[1]),float(bins[2])])
  closure_plot_variables.append(var_name)
  closure_plot_binnings.append(bins_float)

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

def ConvertDatasetName(pf_val,ch,ana,subtraction=False):
  name = "dataframes/{}/{}_{}_dataframe.pkl".format(ana,ch,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("||","_or_").replace("deepTauVsJets","dTvJ"))
  if not subtraction:
    return name
  else:
    return name.replace(".pkl","_subtraction.pkl")

def ConvertMCDatasetName(pf_val,proc,ch,ana):
  return "dataframes/{}/{}_{}_mc_{}_dataframe.pkl".format(ana,ch,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("||","_or_").replace("deepTauVsJets","dTvJ"),proc)

############# Assign dataframes #################

### setup pass and fail ###

if args.fail_wp == None:
  FAIL = "(deepTauVsJets_{}_X==0)".format(args.pass_wp)
else:
  FAIL = "(deepTauVsJets_{}_X==1 && deepTauVsJets_{}_X==0)".format(args.fail_wp,args.pass_wp)

PASS = "(deepTauVsJets_{}_X==1)".format(args.pass_wp)


### Get all combination you need to loop through ###

lst_n = []
for ind,i in enumerate(args.channel):
  if i == "t":
    lst_n.append(ind+1)

total_keys = []
for ind,i in enumerate(lst_n):
  for k in list(itertools.combinations(lst_n, ind+1)):
    if (len(k)==1 and ((not only_do_ff) or only_do_ff==k[0])) or (len(k) > 1 and multiple_ff):
      total_keys.append(k)

### Begin load and training loop ###

if not os.path.isdir("BDTs/{}".format(args.analysis)): os.system("mkdir BDTs/{}".format(args.analysis))
if not os.path.isdir("hyperparameters/{}".format(args.analysis)): os.system("mkdir hyperparameters/{}".format(args.analysis))

datasets_created = []
store = {}

### loop through stages ###
#for rwt_key in ["Raw F_{F}","Alternative F_{F}","Correction"]:
for rwt_key in ["All"]:
  tk_variables = []
  for ind, t_ff in enumerate(total_keys):

    print "<< Running for {} >>".format(t_ff)
  
    ### Setup what dataframes you want to use and where ###
    t_ff_string = str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ","")
    sel = MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL)
  
    store[t_ff_string] = ff_ml()
  
    store[t_ff_string].AddDataframes(ConvertDatasetName(sel["Raw F_{F}"]["fail"],args.channel,args.analysis,subtraction=do_subtraction),ConvertDatasetName(sel["Raw F_{F}"]["pass"],args.channel,args.analysis,subtraction=do_subtraction),"Raw F_{F}")
    #if ind + 1 != len(lst_n):
    if not (len(lst_n) > 1 and len(lst_n) == len(t_ff)):
      # Alternative and Correction from data
      store[t_ff_string].AddDataframes(ConvertDatasetName(sel["Alternative F_{F}"]["fail"],args.channel,args.analysis,subtraction=do_subtraction),ConvertDatasetName(sel["Alternative F_{F}"]["pass"],args.channel,args.analysis,subtraction=do_subtraction),"Alternative F_{F}")
      store[t_ff_string].AddDataframes(ConvertDatasetName(sel["Correction"]["fail"],args.channel,args.analysis,subtraction=do_subtraction),ConvertDatasetName(sel["Correction"]["pass"],args.channel,args.analysis,subtraction=do_subtraction),"Correction")
    else:
      for proc in mc_procs:
        # Alternatice and Correction from MC
        store[t_ff_string].AddDataframes(ConvertMCDatasetName(sel["Raw F_{F}"]["fail"],proc,args.channel,args.analysis),ConvertMCDatasetName(sel["Raw F_{F}"]["pass"],proc,args.channel,args.analysis),"Alternative F_{F}")
        store[t_ff_string].AddDataframes(ConvertMCDatasetName(sel["Signal"]["fail"],proc,args.channel,args.analysis),ConvertMCDatasetName(sel["Signal"]["pass"],proc,args.channel,args.analysis),"Correction")
  
    ### update variables ###
    fitting_variables_update = []
    for var in fitting_variables:
      if "NUM" in var:
        for i in t_ff:
          fitting_variables_update.append(var.replace("NUM",str(i)))
      elif "ALT" in var:
        for i in tuple(set(lst_n) - set(t_ff)):
          fitting_variables_update.append(var.replace("ALT",str(i)))
      else:
        fitting_variables_update.append(var)

    tk_variables.append(fitting_variables_update) 

    if (not combine_taus) or (combine_taus and ind == 0):
      scoring_variables_update = []
      for var in scoring_variables:
        if "NUM" in var:
          for i in t_ff:
            scoring_variables_update.append(var.replace("NUM",str(i)))
        elif "ALT" in var:
          for i in tuple(set(lst_n) - set(t_ff)):
            scoring_variables_update.append(var.replace("ALT",str(i)))
        else:
          scoring_variables_update.append(var)

    cap_at=None

    if (args.scan_batch_ff and "Correction" in rwt_key) or (args.no_correction and "Correction" in rwt_key): continue

    rwt_key_name = rwt_key.lower().replace(" ","_").replace("_{","").replace("}","")
    if not combine_taus:
      model_name = "{}_{}_{}_{}_{}".format(args.channel,str(args.fail_wp),args.pass_wp,rwt_key.lower().replace(" ","_").replace("_{","").replace("}",""),str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ",""))
    else:
      model_name = "{}_{}_{}_{}_all_taus".format(args.channel,str(args.fail_wp),args.pass_wp,rwt_key.lower().replace(" ","_").replace("_{","").replace("}",""))


    if not ((args.load_models_ff and ("F_{F}" in rwt_key or "All" in rwt_key)) or (args.load_models_correction and "Correction" in rwt_key)):

      rwter = reweighter.reweighter(n_estimators=600, learning_rate=0.03, min_samples_leaf=1200, max_depth=2, loss_regularization=5)
      #rwter = reweighter.reweighter(
      #            n_estimators = 200,
      #            learning_rate = 0.04,
      #            min_samples_leaf = 1200,
      #            max_depth = 2,
      #            loss_regularization = 6.0
      #            )

      if ((not args.collect_scan_batch_ff) and ("F_{F}" in rwt_key or "All" in rwt_key)) or (args.scan_correction and "Correction" in rwt_key):
        if (not combine_taus) or (combine_taus and ind == 0):
          train_fail, wt_train_fail, train_pass, wt_train_pass = store[t_ff_string].GetDataframes(rwt_key,data_type="train",variables=copy.deepcopy(fitting_variables_update))
          test_fail, wt_test_fail, test_pass, wt_test_pass = store[t_ff_string].GetDataframes(rwt_key,data_type="test",variables=copy.deepcopy(fitting_variables_update))
          if combine_taus: 
            save_names = train_fail.columns
        else:
          train_fail_temp, wt_train_fail_temp, train_pass_temp, wt_train_pass_temp = store[t_ff_string].GetDataframes(rwt_key,data_type="train",variables=copy.deepcopy(fitting_variables_update))
          test_fail_temp, wt_test_fail_temp, test_pass_temp, wt_test_pass_temp = store[t_ff_string].GetDataframes(rwt_key,data_type="test",variables=copy.deepcopy(fitting_variables_update))
          replace_dict = {}
          for ind_rn, rn in enumerate(train_fail_temp.columns): replace_dict[rn] = save_names[ind_rn]
          train_fail_temp.rename(columns=replace_dict,inplace=True)
          train_pass_temp.rename(columns=replace_dict,inplace=True)
          test_fail_temp.rename(columns=replace_dict,inplace=True)
          test_pass_temp.rename(columns=replace_dict,inplace=True)

          train_fail = pd.concat([train_fail,train_fail_temp], ignore_index=True, sort=False)
          wt_train_fail = pd.concat([wt_train_fail,wt_train_fail_temp], ignore_index=True, sort=False)
          train_pass = pd.concat([train_pass,train_pass_temp], ignore_index=True, sort=False)
          wt_train_pass = pd.concat([wt_train_pass,wt_train_pass_temp], ignore_index=True, sort=False)
          test_fail = pd.concat([test_fail,test_fail_temp], ignore_index=True, sort=False)
          wt_test_fail = pd.concat([wt_test_fail,wt_test_fail_temp], ignore_index=True, sort=False)
          test_pass = pd.concat([test_pass,test_pass_temp], ignore_index=True, sort=False)
          wt_test_pass = pd.concat([wt_test_pass,wt_test_pass_temp], ignore_index=True, sort=False)

          del train_fail_temp, wt_train_fail_temp, train_pass_temp, wt_train_pass_temp, test_fail_temp, wt_test_fail_temp, test_pass_temp, wt_test_pass_temp 
          gc.collect()

      if combine_taus and ind+1 != len(total_keys): continue

      if (args.scan_ff and ("F_{F}" in rwt_key or "All" in rwt_key)) or (args.scan_correction and "Correction" in rwt_key):
  
        print "<< Running grid search reweighting training for {} >>".format(rwt_key)
        rwter.grid_search(train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid, scoring_variables=scoring_variables_update, cap_at=cap_at)
        with open('hyperparameters/{}/ff_hp_{}.json'.format(args.analysis,model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)
  
      elif (args.scan_batch_ff and ("F_{F}" in rwt_key or "All" in rwt_key)) or (args.scan_batch_correction and "Correction" in rwt_key):
  
        print "<< Batch submitting grid search reweighting training for {} >>".format(rwt_key)
        rwter.grid_search_batch(model_name,train_fail, train_pass, wt_train_fail ,wt_train_pass, test_fail, test_pass, wt_test_fail ,wt_test_pass, param_grid=param_grid, scoring_variables=scoring_variables_update, cap_at=cap_at)
  
      elif (args.collect_scan_batch_ff and ("F_{F}" in rwt_key or "All" in rwt_key)) or (args.collect_scan_batch_correction and "Correction" in rwt_key):
  
        print "<< Collecting reweighting training for {} >>".format(rwt_key)
        rwter = rwter.collect_grid_search_batch(model_name)
        with open('hyperparameters/{}/ff_hp_{}.json'.format(args.analysis,model_name), 'w') as outfile: json.dump(rwter.dump_hyperparameters(), outfile)
      else:
  
        if (args.load_hyperparameters_ff and ("F_{F}" in rwt_key or "All" in rwt_key)) or (args.load_hyperparameters_correction and "Correction" in rwt_key):
          with open('hyperparameters/{}/ff_hp_{}.json'.format(args.analysis,model_name)) as json_file: params = json.load(json_file)
          rwter.set_params(params)

        print "<< Running reweighting training for {} >>".format(rwt_key)
        print train_fail
        rwter.norm_and_fit(train_fail, train_pass, wt_train_fail ,wt_train_pass,cap_at=cap_at)
  
      if not (args.scan_batch_ff or args.scan_batch_correction): pkl.dump(rwter,open("BDTs/{}/ff_{}.pkl".format(args.analysis,model_name), "wb"))
  
    else:

      if combine_taus and ind+1 != len(total_keys): continue
  
      print "<< Loading reweighting training for {} >>".format(rwt_key)
      rwter = pkl.load(open("BDTs/{}/ff_{}.pkl".format(args.analysis,model_name), "rb"))

    if not combine_taus: 
      store[t_ff_string].AddModel(rwter,rwt_key)
    else:
      for ind_am, t_ff_am in enumerate(total_keys):
        t_ff_string_am = str(t_ff_am).replace("(","").replace(")","").replace(",","").replace(" ","")
        store[t_ff_string_am].AddModel(rwter,rwt_key)
        store[t_ff_string_am].model[rwt_key].column_names = pd.Index(tk_variables[ind_am])


  ### Run plotting ###
  if not args.no_plots and not args.only_mc:

    print "<< Running plotting >>"
    for ind, t_ff in enumerate(total_keys):

      t_ff_string = str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ","")

      #logx_lst = ["pt_1","pt_2","m_vis","svfit_mass","mt_tot"]
      logx_lst = ["pt_1","pt_2","pt_3","pt_4","m_vis","svfit_mass","mt_tot","mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34"]
      logy_lst = []

      tp = [None]
      if args.do_train_test_plots: tp += ["train","test"]

      yp = ["all_years"]
      if args.do_year_plots: yp += years


      if rwt_key == "Raw F_{F}" or rwt_key == "All":
        #compare_other_rwt = ["wt_ff_mssm_1","wt_ff_dmbins_1","wt_ff_1"]
        compare_other_rwt = []
      else:
        compare_other_rwt = []

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

          store[t_ff_string].PlotReweightsAndClosure(closure_plot_variables, closure_plot_binnings, rwt_key, yr_sel, data_type=tt, title_left=args.channel, title_right=yr, logx=logx_lst, logy=logy_lst, compare_other_rwt=compare_other_rwt, flat_rwt=True, plot_name="{}_{}_{}_{}_{}".format(tt_name,t_ff_string,rwt_key_name,args.channel,yr))
