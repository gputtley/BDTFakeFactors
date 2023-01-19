from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.ff_ml import ff_ml
from UserCode.BDTFakeFactors.functions import PrintDatasetSummary, MakeSelFromList, MakeSelDict
from UserCode.BDTFakeFactors.batch import CreateBatchJob,SubmitBatchJob
from UserCode.BDTFakeFactors import reweighter
from collections import OrderedDict
from array import array
from UserCode.BDTFakeFactors.plotting import FindRebinning, RebinHist
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import itertools
import os
import json
import copy
import gc
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--analysis',help= 'Analysis to train BDT for', default='4tau')
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mmtt')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='loose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
args = parser.parse_args()

if args.batch:
  cmd = "python scripts/ff_non_closure_uncertainty.py"
  for a in vars(args):
    if a == "batch": continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      cmd += " --{}={}".format(a,getattr(args, a))
  name = "ff_non_closure_uncertainty_job_{}_{}_{}".format(args.channel,args.pass_wp,args.fail_wp)
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  SubmitBatchJob("jobs/"+name+".sh",time=300)
  exit()


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

############ Functions ##########################

def ConvertDatasetName(pf_val,ch,ana,subtraction=False):
  name = "dataframes/{}/{}_{}_dataframe.pkl".format(ana,ch,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("||","_or_").replace("deepTauVsJets","dTvJ"))
  if not subtraction:
    return name
  else:
    return name.replace(".pkl","_subtraction.pkl")

def ConvertMCDatasetName(pf_val,proc,ch,ana):
  return "dataframes/{}/{}_{}_mc_{}_dataframe.pkl".format(ana,ch,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("||","_or_").replace("deepTauVsJets","dTvJ"),proc)

def ConvertTH1ToLambda(hist):
  form = "("
  for i in range(1,hist.GetNbinsX()+1):
    if i == 1 and hist.GetNbinsX() > 1:
      form += "({}*(x<{}))".format(hist.GetBinContent(i),hist.GetBinLowEdge(i+1))
    elif i == 1 and hist.GetNbinsX() == 1:
      form += "{})".format(hist.GetBinContent(i))
    elif i == hist.GetNbinsX():
      form += "+ ({}*(x>={})))".format(hist.GetBinContent(i),hist.GetBinLowEdge(i))
    else:
      form += "+ ({}*((x>={}) & (x<{})))".format(hist.GetBinContent(i),hist.GetBinLowEdge(i),hist.GetBinLowEdge(i+1))
  return form

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

    cap_at=None

    rwt_key_name = rwt_key.lower().replace(" ","_").replace("_{","").replace("}","")
    if not combine_taus:
      model_name = "{}_{}_{}_{}_{}".format(args.channel,str(args.fail_wp),args.pass_wp,rwt_key.lower().replace(" ","_").replace("_{","").replace("}",""),str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ",""))
    else:
      model_name = "{}_{}_{}_{}_all_taus".format(args.channel,str(args.fail_wp),args.pass_wp,rwt_key.lower().replace(" ","_").replace("_{","").replace("}",""))

    rwter = pkl.load(open("BDTs/{}/ff_{}.pkl".format(args.analysis,model_name), "rb"))
    store[t_ff_string].AddModel(rwter,rwt_key)

    if (not combine_taus) or (combine_taus and ind == 0):
      train_fail, wt_train_fail, train_pass, wt_train_pass = store[t_ff_string].GetDataframes(rwt_key,data_type="train",variables=copy.deepcopy(fitting_variables_update),with_reweights=True)
      test_fail, wt_test_fail, test_pass, wt_test_pass = store[t_ff_string].GetDataframes(rwt_key,data_type="test",variables=copy.deepcopy(fitting_variables_update),with_reweights=True)
      if combine_taus: 
        save_names = train_fail.columns
    else:
      store[t_ff_string].model[rwt_key].column_names = pd.Index(fitting_variables_update)
      train_fail_temp, wt_train_fail_temp, train_pass_temp, wt_train_pass_temp = store[t_ff_string].GetDataframes(rwt_key,data_type="train",variables=copy.deepcopy(fitting_variables_update),with_reweights=True)
      test_fail_temp, wt_test_fail_temp, test_pass_temp, wt_test_pass_temp = store[t_ff_string].GetDataframes(rwt_key,data_type="test",variables=copy.deepcopy(fitting_variables_update),with_reweights=True)
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

    if not combine_taus or (combine_taus and ind + 1 == len(total_keys)):
      rename_cols = {}
      for col in test_fail.columns:
        rename_cols[col] = col.replace("(","_").replace("*","_times_").replace("/","_divide_").replace(")","")

      df_write_pass = Dataframe()
      train_pass = train_pass.rename(columns=rename_cols)
      train_pass.loc[:,"weights"] = wt_train_pass
      df_write_pass.dataframe = copy.deepcopy(train_pass)
      ttree_pass = df_write_pass.WriteToRoot(None,key='ntuple',return_tree=True)

      df_write_fail = Dataframe()
      train_fail = train_fail.rename(columns=rename_cols)
      train_fail.loc[:,"weights"] = wt_train_fail
      df_write_fail.dataframe = copy.deepcopy(train_fail)
      ttree_fail = df_write_fail.WriteToRoot(None,key='ntuple',return_tree=True)
 
      l_dump = OrderedDict()
      for col in test_fail.columns:

        n_bins = 20
        ignore_quantile = 0.01

        var_name = col.replace("(","_").replace("*","_times_").replace("/","_divide_").replace(")","")

        concat_test = pd.concat([test_pass,test_fail],axis=0,ignore_index=True)
        if len(concat_test.loc[:,col].unique()) > n_bins: # continuous
          bins = list(np.linspace(concat_test.loc[:,col].quantile(ignore_quantile),concat_test.loc[:,col].quantile(1-ignore_quantile),n_bins))
          bins[0] = concat_test.loc[:,col].min()
          bins[-1] = concat_test.loc[:,col].max()
        else:
          bins = sorted(concat_test.loc[:,col].unique())
          bins.append(2*bins[-1]-bins[-2])
        del concat_test
        gc.collect()

        bins = array('f', map(float,bins))

        hout = ROOT.TH1D('hout','',len(bins)-1, bins)
        for i in range(0,hout.GetNbinsX()+2): hout.SetBinError(i,0)
        hout.SetDirectory(ROOT.gROOT)

        hist_fail = hout.Clone()
        hist_fail.SetName('hist_fail')
        hist_fail.SetDirectory(ROOT.gROOT)
        ttree_fail.Draw('%(var_name)s>>hist_fail' % vars(),'weights*(1)','goff')
        hist_fail = ttree_fail.GetHistogram() 

        hist_pass = hout.Clone()
        hist_pass.SetName('hist_pass')
        hist_pass.SetDirectory(ROOT.gROOT)
        ttree_pass.Draw('%(var_name)s>>hist_pass' % vars(),'weights*(1)','goff')
        hist_pass = ttree_pass.GetHistogram()

        binning = FindRebinning(hist_fail,BinThreshold=100,BinUncertFraction=0.5)
        hist_pass = RebinHist(hist_pass,binning)
        binning = FindRebinning(hist_pass,BinThreshold=100,BinUncertFraction=0.5)
        hist_pass = RebinHist(hist_pass,binning)
        hist_fail = RebinHist(hist_fail,binning)

        neg_hist_fail = hist_fail.Clone()
        neg_hist_fail.Scale(-1)
        hist_pass.Add(neg_hist_fail.Clone())
        hist_pass.Divide(hist_fail.Clone())
        for i in range(0,hist_pass.GetNbinsX()+1): hist_pass.SetBinContent(i,abs(hist_pass.GetBinContent(i)))
        l_dump[col] = ConvertTH1ToLambda(hist_pass)

      with open('BDTs/ff_non_closure_uncertainties_{}_{}.json'.format(args.analysis,args.channel), 'w') as outfile: json.dump(l_dump, outfile, indent=2)
