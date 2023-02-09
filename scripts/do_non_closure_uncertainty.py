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
import yaml
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',help= 'Config to input', default='mmtt.yaml')
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
args = parser.parse_args()

############ Import config ####################

with open(args.cfg) as f: data = yaml.load(f)

############ Batch Submission ####################

if args.batch:
  cmd = "python scripts/do_non_closure_uncertainty.py"
  for a in vars(args):
    if a == "batch": continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      cmd += " --{}={}".format(a,getattr(args, a))
  name = "non_closure_job_{}".format(data["channel"])
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  #SubmitBatchJob("jobs/"+name+".sh",time=300)
  SubmitBatchJob("jobs/"+name+".sh")
  exit()

############ Functions ##########################

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
rwter = pkl.load(open("BDTs/{}/ff_{}.pkl".format(data["channel"],model_name), "rb"))
store.AddModel(rwter,"All")
train_fail, wt_train_fail, train_pass, wt_train_pass = store.GetDataframes("All",data_type="train",with_reweights=True)
test_fail, wt_test_fail, test_pass, wt_test_pass = store.GetDataframes("All",data_type="test",with_reweights=True)

df_write_pass = Dataframe()
test_pass.loc[:,"weights"] = wt_test_pass
df_write_pass.dataframe = copy.deepcopy(test_pass)
ttree_pass = df_write_pass.WriteToRoot(None,key='ntuple',return_tree=True)

df_write_fail = Dataframe()
test_fail.loc[:,"weights"] = wt_test_fail
df_write_fail.dataframe = copy.deepcopy(test_fail)
ttree_fail = df_write_fail.WriteToRoot(None,key='ntuple',return_tree=True)

l_dump = OrderedDict()
for col in test_fail.columns:

  if "weights" in col: continue

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

with open('BDTs/ff_non_closure_uncertainties_{}.json'.format(data["channel"]), 'w') as outfile: json.dump(l_dump, outfile, indent=2)
