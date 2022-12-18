import uproot
import os
import json
import pandas
import pickle
import argparse
import numpy as np
import xgboost as xgb
import itertools
import copy
from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.reweighter import reweighter
from UserCode.BDTFakeFactors.functions import SampleValuesAndGiveLargestShift, GetNonClosureLargestShift

# python scripts/add_ff_to_trees.py --input_location=/vols/cms/gu18/Offline/output/4tau/2018_1907 --output_location="./" --filename=TauB_mmtt_2018.root --channel=mmtt --year=2018 --splitting=100000 --offset=0

parser = argparse.ArgumentParser()
parser.add_argument('--analysis',help= 'Analysis to train BDT for', default='4tau')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='vvvloose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--input_location',help= 'Name of input location (not including file name)', default='/vols/cms/gu18/Offline/output/4tau/2018_1907')
parser.add_argument('--output_location',help= 'Name of output location (not including file name)', default='./')
parser.add_argument('--filename',help= 'Name of file', default='TauB_tttt_2018.root')
parser.add_argument('--channel',help= 'Name of channel', default='tttt')
parser.add_argument('--year',help= 'Name of year', default='2018')
parser.add_argument('--splitting',help= 'Number of events per task', default='100000')
parser.add_argument('--offset',help= 'Offset of job', default='0')
args = parser.parse_args()

with open("input/{}/{}/config.json".format(args.analysis,args.channel)) as jsonfile:
  config = json.load(jsonfile)

years = config["years"]
SIDEBAND = config["DR"]
SIGNAL = config["SR"]
mc_procs = config["mc_procs"]
multiple_ff = config["multiple_ff"]
only_do_ff = config["only_do_ff"]

lst_n = []
for ind,i in enumerate(args.channel):
  if i == "t":
    lst_n.append(ind+1)

total_keys = []
for ind,i in enumerate(lst_n):
  for k in list(itertools.combinations(lst_n, ind+1)):
    if (len(k)==1 and ((not only_do_ff) or only_do_ff==k[0])) or (len(k) > 1 and multiple_ff):
      total_keys.append(k)

tree = uproot.open(args.input_location+'/'+args.filename, localsource=uproot.FileSource.defaults)["ntuple"]

models_to_add = {}

for ind, t_ff in enumerate(total_keys):
  str_k = ""
  for ks in t_ff: str_k += str(ks)
  models_to_add[str_k] = {} 
  models_to_add[str_k]["wt_ff_ml_{}".format(str_k)] = "BDTs/{}/ff_{}_{}_{}_all_all_taus.pkl".format(args.analysis,args.channel,args.fail_wp,args.pass_wp)
  if ind == 0:
    xgb_model_var = list(pickle.load(open(models_to_add[str_k]["wt_ff_ml_{}".format(str_k)], "rb")).column_names)

lowest_end = 10
highest_end = 0
for i in xgb_model_var:
  if i[-1].isdigit() and i[-2] == "_":
    if int(i[-1]) < lowest_end: lowest_end = int(i[-1])
    if int(i[-1]) > highest_end: highest_end = int(i[-1])

def MakeReplaceDict(low,high,val):
  initial_lst = range(low,high+1)
  initial = ""
  for i in initial_lst:
    initial += str(i)
  final = initial.replace(val,"")
  final = val + final
  rd = {}
  for ind,s in enumerate(initial): rd[s] = final[ind]
  return rd

def ReplaceNumber(string,low,high,val):
  rd = MakeReplaceDict(low,high,val)
  for k,v in rd.items(): 
    if string[:2] != "yr":
      string = string.replace("_"+k,"_X"+k)

  for k,v in rd.items(): 
    if string[:2] != "yr":
      string = string.replace("_X"+k,"_"+v)
  return string

k = 0
for small_tree in tree.iterate(entrysteps=int(args.splitting)):
  print k
  if k == int(args.offset) or int(args.offset)==-1:
    df = Dataframe()
    df.dataframe = pandas.DataFrame.from_dict(small_tree)
    print "Loaded tree"

    # add independent weights
    for mkey, mval in models_to_add.items():
      for key, val in mval.items():
        xgb_model = pickle.load(open(val, "rb"))       
        new_df = Dataframe()
        features = list(xgb_model.column_names)
        for yr in years: features.remove("yr_"+yr)
        new_df.dataframe = df.dataframe.copy(deep=False)
        
         
        if len(models_to_add) > 1:
          for ind, i in enumerate(features): features[ind] = ReplaceNumber(features[ind],lowest_end,highest_end,mkey[-1])

        new_df.SelectColumns(features)
        for yr in years: new_df.dataframe.loc[:,"yr_"+yr] = (args.year==yr)
        
        total_features = list(xgb_model.column_names)

        if len(models_to_add) > 1:       
          for ind, i in enumerate(total_features): total_features[ind] = ReplaceNumber(total_features[ind],lowest_end,highest_end,mkey[-1])

        new_df.dataframe = new_df.dataframe.reindex(total_features, axis=1)
        df.dataframe.loc[:,key] = xgb_model.predict_reweights(new_df.dataframe) 

        # uncertainties

        non_closure_uncert = GetNonClosureLargestShift(copy.deepcopy(new_df.dataframe),"BDTs/ff_non_closure_uncertainties_{}_{}.json".format(args.analysis, args.channel),df.dataframe.loc[:,key])
        df.dataframe.loc[:,key+"_non_closure_up"] = non_closure_uncert.loc[:,"up"]
        df.dataframe.loc[:,key+"_non_closure_down"] = non_closure_uncert.loc[:,"down"]
        if args.channel != "ttt":
          pass_q = [0.0]
          fail_q = [-4.0,-2.0,0.0,2.0,4.0]
        else:
          pass_q = [-1.0,1.0]
          fail_q = [-3.0,3.0]
        q_sum_uncert = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,"q_sum",pass_q,fail_q,continuous=False,n_samples=10)
        df.dataframe.loc[:,key+"_q_sum_up"] = q_sum_uncert.loc[:,"up"]
        df.dataframe.loc[:,key+"_q_sum_down"] = q_sum_uncert.loc[:,"down"]
        iso_features = [x for x in new_df.dataframe.columns if "deepTauVsJets_iso_" in x]
        iso_uncert = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,iso_features,[[0.1,1.0]]*len(iso_features),[[0.1,1.0]]*len(iso_features),continuous=True,n_samples=50)
        df.dataframe.loc[:,key+"_iso_up"] = iso_uncert.loc[:,"up"]
        df.dataframe.loc[:,key+"_iso_down"] = iso_uncert.loc[:,"down"]

        #print df.dataframe.loc[:,[key,key+"_non_closure_up",key+"_non_closure_down",key+"_q_sum_up",key+"_q_sum_down",key+"_iso_up",key+"_iso_down"]]
        #print df.dataframe
    x = df.WriteToRoot(args.output_location+'/'+args.filename.replace(".root","_"+str(k)+".root"),return_tree=False)
    del df, small_tree
    if int(args.offset)!=-1: break
  k += 1
print "Finished processing"
