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
import re
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

#with open("input/{}/{}/config.json".format(args.analysis,args.channel)) as jsonfile:
#  config = json.load(jsonfile)

years = ["2016_preVFP","2016_postVFP","2017","2018"]
#mc_procs = config["mc_procs"]
#multiple_ff = config["multiple_ff"]
#only_do_ff = config["only_do_ff"]
multiple_ff = False
only_do_ff = False


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
  #models_to_add[str_k]["wt_ff_ml_{}".format(str_k)] = "BDTs/{}/ff_{}_{}_{}_all_all_taus.pkl".format(args.analysis,args.channel,args.fail_wp,args.pass_wp)
  if args.channel != "tttt":
    models_to_add[str_k]["wt_ff_ml_{}".format(str_k)] = "BDTs/{}/ff_reweights_{}.pkl".format(args.channel,args.channel)
  else:
    models_to_add[str_k]["wt_ff_ml_{}".format(str_k)] = "BDTs/ttt/ff_reweights_ttt.pkl"
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

#def ReplaceNumber(string,low,high,val):
#  rd = MakeReplaceDict(low,high,val)
#  for k,v in rd.items(): 
#    if string[:2] != "yr":
#      string = string.replace("_"+k,"_X"+k)
#
#  for k,v in rd.items(): 
#    if string[:2] != "yr":
#      string = string.replace("_X"+k,"_"+v)
#  return string

def ReplaceNumber(keys,num,other):
  other.sort()
  change_dict = {"1":str(num)}
  for ind,i in enumerate(other):
    change_dict[str(ind+2)] = str(i)

  replace_dict = {}
  for k in keys:
    replace_dict[k] = k
    for kc, vc in change_dict.items():
      if kc in k:
        replace_dict[k] = replace_dict[k].replace(kc,vc)
        break
  return replace_dict.values()


k = 0
no_loop = True
for small_tree in tree.iterate(entrysteps=int(args.splitting)):
  if k == int(args.offset) or int(args.offset)==-1:
    no_loop = False
    df = Dataframe()
    df.dataframe = pandas.DataFrame.from_dict(small_tree)
    print "Loaded tree"

    # add independent weights
    for mkey, mval in models_to_add.items():
      for key, val in mval.items():

        xgb_model = pickle.load(open(val, "rb"))       
        new_df = Dataframe()
        features = list(xgb_model.column_names)
        #features = [i.replace("fabs_","") for i in features]
        for yr in years: features.remove("yr_"+yr)
        features.remove("tau_number")
        new_df.dataframe = df.dataframe.copy(deep=False)
      

        other = [int(m.start()+1) for m in re.finditer('t', args.channel)]
        other.remove(int(mkey))
        print "------------------------------------------------------------------------------"
        if args.channel == "tttt": 
          lowest_pT = max(other)
          #print other, lowest_pT
          other.remove(lowest_pT) 
          new_df.SelectColumns(ReplaceNumber(features,mkey,other)+["q_{}".format(lowest_pT)]) 
          #print new_df.dataframe
          new_df.dataframe.loc[:,"q_sum"] = new_df.dataframe.loc[:,"q_sum"] - new_df.dataframe.loc[:,"q_{}".format(lowest_pT)]
          #print new_df.dataframe
          new_df.dataframe = new_df.dataframe.drop(["q_{}".format(lowest_pT)],axis=1) 
          #print new_df.dataframe
        else:
          #print new_df
          #print ReplaceNumber(features,mkey,other)
          new_df.SelectColumns(ReplaceNumber(features,mkey,other)) 

        for yr in years: new_df.dataframe.loc[:,"yr_"+yr] = (args.year==yr)
        new_df.dataframe.loc[:,"tau_number"] = int(mkey)
        if args.channel == "tttt": new_df.dataframe.loc[(new_df.dataframe.loc[:,"tau_number"]==4),"tau_number"] = 3
        for i in list(new_df.dataframe.columns): 
          if "fabs(" in i:
            new_df.dataframe = new_df.dataframe.rename(columns={i:i.replace("fabs(","fabs_").replace(")","")})
        new_df.RenumberColumns(int(mkey))

        total_features = list(xgb_model.column_names)
        total_features = [i.replace("(","_").replace(")","") for i in total_features]
        new_df.dataframe = new_df.dataframe.reindex(total_features, axis=1)
        df.dataframe.loc[:,key] = xgb_model.predict_reweights(new_df.dataframe) 

        # uncertainties
        if args.channel != "tttt":
          non_closure_uncert = GetNonClosureLargestShift(copy.deepcopy(new_df.dataframe),"BDTs/ff_non_closure_uncertainties_{}.json".format(args.channel),df.dataframe.loc[:,key],name="non_closure")
          subtract_pass_non_closure_uncert = GetNonClosureLargestShift(copy.deepcopy(new_df.dataframe),"BDTs/subtraction_non_closure_uncertainties_pass_{}.json".format(args.channel),df.dataframe.loc[:,key],"subtract_pass_non_closure")
          subtract_fail_non_closure_uncert = GetNonClosureLargestShift(copy.deepcopy(new_df.dataframe),"BDTs/subtraction_non_closure_uncertainties_fail_{}.json".format(args.channel),df.dataframe.loc[:,key],"subtract_fail_non_closure")
        else:
          non_closure_uncert = GetNonClosureLargestShift(copy.deepcopy(new_df.dataframe),"BDTs/ff_non_closure_uncertainties_ttt.json",df.dataframe.loc[:,key],name="non_closure")
          subtract_pass_non_closure_uncert = GetNonClosureLargestShift(copy.deepcopy(new_df.dataframe),"BDTs/subtraction_non_closure_uncertainties_pass_ttt.json",df.dataframe.loc[:,key],name="subtract_pass_non_closure")
          subtract_fail_non_closure_uncert = GetNonClosureLargestShift(copy.deepcopy(new_df.dataframe),"BDTs/subtraction_non_closure_uncertainties_fail_ttt.json",df.dataframe.loc[:,key],name="subtract_fail_non_closure")

        df.dataframe.loc[:,key+"_non_closure_up"] = non_closure_uncert.loc[:,"up"]
        df.dataframe.loc[:,key+"_non_closure_down"] = non_closure_uncert.loc[:,"down"]

        df.dataframe.loc[:,key+"_subtract_pass_non_closure_up"] = subtract_pass_non_closure_uncert.loc[:,"up"]
        df.dataframe.loc[:,key+"_subtract_pass_non_closure_down"] = subtract_pass_non_closure_uncert.loc[:,"down"]

        df.dataframe.loc[:,key+"_subtract_fail_non_closure_up"] = subtract_fail_non_closure_uncert.loc[:,"up"]
        df.dataframe.loc[:,key+"_subtract_fail_non_closure_down"] = subtract_fail_non_closure_uncert.loc[:,"down"]

        if args.channel not in ["ttt","tttt"]:
          pass_q = [0.0]
          fail_q = [-4.0,-2.0,2.0,4.0]
        else:
          pass_q = [-1.0,1.0]
          fail_q = [-3.0,3.0]

        iso_features = [x for x in new_df.dataframe.columns if "deepTauVsJets_loose_" in x and "_1" not in x]
        fail_iso = [[0.0,1.0]]*len(iso_features)

        tau_numbers = [float(ind+1) for ind, i in enumerate(args.channel) if "t" in i]
        tau_numbers.remove(int(mkey))

        #if len(iso_features) == 1:
        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[1.0],[float(mkey)]]
        #  fail_sel = fail_iso + [[float(mkey)]]
        #  iso_uncert_0f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="iso_uncert_{}_0f".format(mkey))
        #  df.dataframe.loc[:,key+"_iso_0f_{}_up".format(mkey)] = iso_uncert_0f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_iso_0f_{}_down".format(mkey)] = iso_uncert_0f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number", "q_sum"]
        #  pass_sel = [[1.0],[float(mkey)],pass_q+fail_q]
        #  fail_sel = [[1.0],[float(mkey)],pass_q+fail_q]
        #  q_sum_uncert_0f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="q_sum_uncert_{}_0f".format(mkey))
        #  df.dataframe.loc[:,key+"_q_sum_0f_{}_up".format(mkey)] = q_sum_uncert_0f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_q_sum_0f_{}_down".format(mkey)] = q_sum_uncert_0f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[1.0],[float(mkey)]]
        #  fail_sel = [[1.0],tau_numbers]
        #  tau_number_uncert_0f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="tau_number_uncert_{}_0f".format(mkey))
        #  df.dataframe.loc[:,key+"_tau_number_0f_{}_up".format(mkey)] = tau_number_uncert_0f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_tau_number_0f_{}_down".format(mkey)] = tau_number_uncert_0f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[0.0],[float(mkey)]]
        #  fail_sel = fail_iso + [[float(mkey)]]
        #  iso_uncert_1f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="iso_uncert_{}_1f".format(mkey))
        #  df.dataframe.loc[:,key+"_iso_1f_{}_up".format(mkey)] = iso_uncert_1f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_iso_1f_{}_down".format(mkey)] = iso_uncert_1f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number", "q_sum"]
        #  pass_sel = [[0.0],[float(mkey)],pass_q+fail_q]
        #  fail_sel = [[0.0],[float(mkey)],pass_q+fail_q]
        #  q_sum_uncert_1f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="q_sum_uncert_{}_1f".format(mkey))
        #  df.dataframe.loc[:,key+"_q_sum_1f_{}_up".format(mkey)] = q_sum_uncert_1f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_q_sum_1f_{}_down".format(mkey)] = q_sum_uncert_1f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[0.0],[float(mkey)]]
        #  fail_sel = [[0.0],tau_numbers]
        #  tau_number_uncert_1f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="tau_number_uncert_{}_1f".format(mkey))
        #  df.dataframe.loc[:,key+"_tau_number_1f_{}_up".format(mkey)] = tau_number_uncert_1f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_tau_number_1f_{}_down".format(mkey)] = tau_number_uncert_1f.loc[:,"down"]

        #elif len(iso_features) == 2:
        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[1.0],[1.0],[float(mkey)]]
        #  fail_sel = fail_iso + [[float(mkey)]]
        #  iso_uncert_0f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="iso_uncert_{}_0f".format(mkey))
        #  df.dataframe.loc[:,key+"_iso_0f_{}_up".format(mkey)] = iso_uncert_0f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_iso_0f_{}_down".format(mkey)] = iso_uncert_0f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number", "q_sum"]
        #  pass_sel = [[1.0],[1.0],[float(mkey)],pass_q+fail_q]
        #  fail_sel = [[1.0],[1.0],[float(mkey)],pass_q+fail_q]
        #  q_sum_uncert_0f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="q_sum_uncert_{}_0f".format(mkey))
        #  df.dataframe.loc[:,key+"_q_sum_0f_{}_up".format(mkey)] = q_sum_uncert_0f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_q_sum_0f_{}_down".format(mkey)] = q_sum_uncert_0f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[1.0],[1.0],[float(mkey)]]
        #  fail_sel = [[1.0],[1.0],tau_numbers]
        #  tau_number_uncert_0f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="tau_number_uncert_{}_0f".format(mkey))
        #  df.dataframe.loc[:,key+"_tau_number_0f_{}_up".format(mkey)] = tau_number_uncert_0f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_tau_number_0f_{}_down".format(mkey)] = tau_number_uncert_0f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[0.0],[1.0],[float(mkey)]]
        #  fail_sel = fail_iso + [[float(mkey)]]
        #  iso_uncert_1f1p = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="iso_uncert_{}_1f1p".format(mkey))
        #  df.dataframe.loc[:,key+"_iso_1f1p_{}_up".format(mkey)] = iso_uncert_1f1p.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_iso_1f1p_{}_down".format(mkey)] = iso_uncert_1f1p.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number", "q_sum"]
        #  pass_sel = [[0.0],[1.0],[float(mkey)],pass_q+fail_q]
        #  fail_sel = [[0.0],[1.0],[float(mkey)],pass_q+fail_q]
        #  q_sum_uncert_1f1p = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="q_sum_uncert_{}_1f1p".format(mkey))
        #  df.dataframe.loc[:,key+"_q_sum_1f1p_{}_up".format(mkey)] = q_sum_uncert_1f1p.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_q_sum_1f1p_{}_down".format(mkey)] = q_sum_uncert_1f1p.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[0.0],[1.0],[float(mkey)]]
        #  fail_sel = [[0.0],[1.0],tau_numbers]
        #  tau_number_uncert_1f1p = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="tau_number_uncert_{}_1f1p".format(mkey))
        #  df.dataframe.loc[:,key+"_tau_number_1f1p_{}_up".format(mkey)] = tau_number_uncert_1f1p.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_tau_number_1f1p_{}_down".format(mkey)] = tau_number_uncert_1f1p.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[1.0],[0.0],[float(mkey)]]
        #  fail_sel = fail_iso + [[float(mkey)]]
        #  iso_uncert_1p1f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="iso_uncert_{}_1p1f".format(mkey))
        #  df.dataframe.loc[:,key+"_iso_1p1f_{}_up".format(mkey)] = iso_uncert_1p1f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_iso_1p1f_{}_down".format(mkey)] = iso_uncert_1p1f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number", "q_sum"]
        #  pass_sel = [[1.0],[0.0],[float(mkey)],pass_q+fail_q]
        #  fail_sel = [[1.0],[0.0],[float(mkey)],pass_q+fail_q]
        #  q_sum_uncert_1p1f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="q_sum_uncert_{}_1p1f".format(mkey))
        #  df.dataframe.loc[:,key+"_q_sum_1p1f_{}_up".format(mkey)] = q_sum_uncert_1p1f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_q_sum_1p1f_{}_down".format(mkey)] = q_sum_uncert_1p1f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[1.0],[0.0],[float(mkey)]]
        #  fail_sel = [[1.0],[0.0],tau_numbers]
        #  tau_number_uncert_1p1f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="tau_number_uncert_{}_1p1f".format(mkey))
        #  df.dataframe.loc[:,key+"_tau_number_1p1f_{}_up".format(mkey)] = tau_number_uncert_1p1f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_tau_number_1p1f_{}_down".format(mkey)] = tau_number_uncert_1p1f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[0.0],[0.0],[float(mkey)]]
        #  fail_sel = fail_iso + [[float(mkey)]]
        #  iso_uncert_2f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="iso_uncert_{}_2f".format(mkey))
        #  df.dataframe.loc[:,key+"_iso_2f_{}_up".format(mkey)] = iso_uncert_2f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_iso_2f_{}_down".format(mkey)] = iso_uncert_2f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number", "q_sum"]
        #  pass_sel = [[0.0],[0.0],[float(mkey)],pass_q+fail_q]
        #  fail_sel = [[0.0],[0.0],[float(mkey)],pass_q+fail_q]
        #  q_sum_uncert_2f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="q_sum_uncert_{}_2f".format(mkey))
        #  df.dataframe.loc[:,key+"_q_sum_2f_{}_up".format(mkey)] = q_sum_uncert_2f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_q_sum_2f_{}_down".format(mkey)] = q_sum_uncert_2f.loc[:,"down"]

        #  uncert_features = iso_features + ["tau_number"]
        #  pass_sel = [[0.0],[0.0],[float(mkey)]]
        #  fail_sel = [[0.0],[0.0],tau_numbers]
        #  tau_number_uncert_2f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,uncert_features,pass_sel,fail_sel,continuous=False,n_samples=10,name="tau_number_uncert_{}_2f".format(mkey))
        #  df.dataframe.loc[:,key+"_tau_number_2f_{}_up".format(mkey)] = tau_number_uncert_2f.loc[:,"up"]
        #  df.dataframe.loc[:,key+"_tau_number_2f_{}_down".format(mkey)] = tau_number_uncert_2f.loc[:,"down"]


        #q_sum_uncert = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,"q_sum",pass_q,fail_q,continuous=False,n_samples=10)
        #df.dataframe.loc[:,key+"_q_sum_up"] = q_sum_uncert.loc[:,"up"]
        #df.dataframe.loc[:,key+"_q_sum_down"] = q_sum_uncert.loc[:,"down"]

        iso_features = [x for x in new_df.dataframe.columns if "deepTauVsJets_loose_" in x and "_1" not in x]
        fail_iso = [[0.0,1.0]]*len(iso_features)
        
        iso_features.append("q_sum")
        fail_iso.append(fail_q+pass_q)

        tau_numbers = [float(ind+1) for ind, i in enumerate(args.channel) if "t" in i]
        tn = int(mkey)
        if args.channel == "tttt": 
          if tn == 4: tn = 3
          tau_numbers.remove(4)
        tau_numbers.remove(tn)
        iso_features.append("tau_number")
        fail_iso.append(tau_numbers)

        if len(iso_features) == 3:
          iso_uncert_0f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,iso_features,[[1.0],fail_q+pass_q,[tn]],fail_iso,continuous=False,n_samples=40,name="iso_uncert_{}_0f".format(mkey))
          df.dataframe.loc[:,key+"_iso_0f_{}_up".format(mkey)] = iso_uncert_0f.loc[:,"up"]
          df.dataframe.loc[:,key+"_iso_0f_{}_down".format(mkey)] = iso_uncert_0f.loc[:,"down"]
          iso_uncert_1f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,iso_features,[[0.0],fail_q+pass_q,[tn]],fail_iso,continuous=False,n_samples=40,name="iso_uncert_{}_1f".format(mkey))
          df.dataframe.loc[:,key+"_iso_1f_{}_up".format(mkey)] = iso_uncert_1f.loc[:,"up"]
          df.dataframe.loc[:,key+"_iso_1f_{}_down".format(mkey)] = iso_uncert_1f.loc[:,"down"]
        elif len(iso_features) == 4:
          print iso_features
          iso_uncert_0f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,iso_features,[[1.0],[1.0],fail_q+pass_q,[tn]],fail_iso,continuous=False,n_samples=40,name="iso_uncert_{}_0f".format(mkey))
          df.dataframe.loc[:,key+"_iso_0f_{}_up".format(mkey)] = iso_uncert_0f.loc[:,"up"]
          df.dataframe.loc[:,key+"_iso_0f_{}_down".format(mkey)] = iso_uncert_0f.loc[:,"down"]
          iso_uncert_1f1p = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,iso_features,[[0.0],[1.0],fail_q+pass_q,[tn]],fail_iso,continuous=False,n_samples=40,name="iso_uncert_{}_1f1p".format(mkey))
          df.dataframe.loc[:,key+"_iso_1f1p_{}_up".format(mkey)] = iso_uncert_1f1p.loc[:,"up"]
          df.dataframe.loc[:,key+"_iso_1f1p_{}_down".format(mkey)] = iso_uncert_1f1p.loc[:,"down"]
          iso_uncert_1p1f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,iso_features,[[1.0],[0.0],fail_q+pass_q,[tn]],fail_iso,continuous=False,n_samples=40,name="iso_uncert_{}_1p1f".format(mkey))
          df.dataframe.loc[:,key+"_iso_1p1f_{}_up".format(mkey)] = iso_uncert_1p1f.loc[:,"up"]
          df.dataframe.loc[:,key+"_iso_1p1f_{}_down".format(mkey)] = iso_uncert_1p1f.loc[:,"down"]
          iso_uncert_2f = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,iso_features,[[0.0],[0.0],fail_q+pass_q,[tn]],fail_iso,continuous=False,n_samples=40,name="iso_uncert_{}_2f".format(mkey))
          df.dataframe.loc[:,key+"_iso_2f_{}_up".format(mkey)] = iso_uncert_2f.loc[:,"up"]
          df.dataframe.loc[:,key+"_iso_2f_{}_down".format(mkey)] = iso_uncert_2f.loc[:,"down"]

        xgb_model_up = pickle.load(open(val.replace(".pkl","_up.pkl"), "rb"))
        xgb_model_down = pickle.load(open(val.replace(".pkl","_down.pkl"), "rb"))
        df.dataframe.loc[:,key+"_subtraction_tmp_up"] = xgb_model_up.predict_reweights(new_df.dataframe)
        df.dataframe.loc[:,key+"_subtraction_tmp_down"] = xgb_model_down.predict_reweights(new_df.dataframe)
        df.dataframe.loc[:,key+"_subtraction_tmp_up"] = df.dataframe.loc[:,key+"_subtraction_tmp_up"].divide(df.dataframe.loc[:,key])
        df.dataframe.loc[:,key+"_subtraction_tmp_down"] = df.dataframe.loc[:,key+"_subtraction_tmp_down"].divide(df.dataframe.loc[:,key])
        df.dataframe.loc[(df.dataframe.loc[:,key+"_subtraction_tmp_up"]<1.0),key+"_subtraction_tmp_up"] = 1.0/df.dataframe.loc[:,key+"_subtraction_tmp_up"]
        df.dataframe.loc[(df.dataframe.loc[:,key+"_subtraction_tmp_down"]<1.0),key+"_subtraction_tmp_down"] = 1.0/df.dataframe.loc[:,key+"_subtraction_tmp_down"]

        df.dataframe.loc[:,key+"_subtraction_up"] = df.dataframe.loc[:,[key+"_subtraction_tmp_up",key+"_subtraction_tmp_down"]].max(axis=1)
        df.dataframe.loc[:,key+"_subtraction_down"] = df.dataframe.loc[:,key+"_subtraction_up"]**(-1)
        df.dataframe.loc[:,key+"_subtraction_up"] = df.dataframe.loc[:,key+"_subtraction_up"].multiply(df.dataframe.loc[:,key])
        df.dataframe.loc[:,key+"_subtraction_down"] = df.dataframe.loc[:,key+"_subtraction_down"].multiply(df.dataframe.loc[:,key])

        #xgb_model_down = pickle.load(open(val.replace(".pkl","_down.pkl"), "rb"))
        #df.dataframe.loc[:,key+"_subtraction_down"] = xgb_model_down.predict_reweights(new_df.dataframe)

        print "Subtraction Up Ave Shift:", df.dataframe.loc[:,key+"_subtraction_up"].sum()/df.dataframe.loc[:,key].sum()
        print "Subtraction Down Ave Shift:", df.dataframe.loc[:,key+"_subtraction_down"].sum()/df.dataframe.loc[:,key].sum()

        #tau_number = SampleValuesAndGiveLargestShift(copy.deepcopy(new_df.dataframe),xgb_model,["tau_number"],[int(mkey)],[tau_numbers],continuous=False,n_samples=10,name="tau_number_{}".format(mkey))
        #df.dataframe.loc[:,key+"_tau_number_{}_up".format(mkey)] = tau_number.loc[:,"up"]
        #df.dataframe.loc[:,key+"_tau_number_{}_down".format(mkey)] = tau_number.loc[:,"down"]

    df.WriteToRoot(args.output_location+'/'+args.filename.replace(".root","_"+str(k)+".root"),return_tree=False)
    del df, small_tree
    if int(args.offset)!=-1: break
  k += 1

if no_loop and k == 0:
  import ROOT
  from array import array
  f = ROOT.TFile(args.input_location+'/'+args.filename)
  t = f.Get("ntuple")

  nf = ROOT.TFile(args.output_location+'/'+args.filename.replace(".root","_"+str(k)+".root"),"RECREATE")
  print "Created", args.output_location+'/'+args.filename.replace(".root","_"+str(k)+".root")
  nt = ROOT.TTree()
  empty = array( 'i', [ ] )
  for i in t.GetListOfBranches():
    nt.Branch(i.GetName(), empty, i.GetName()+'/I')

  for mkey, mval in models_to_add.items():
    for key, val in mval.items():
      nt.Branch(key, empty, key+'/I')
      nt.Branch(key+"_non_closure_up", empty, key+"_non_closure_up"+'/I')
      nt.Branch(key+"_non_closure_down", empty, key+"_non_closure_down"+'/I')
      nt.Branch(key+"_q_sum_up", empty, key+"_q_sum_up"+'/I')
      nt.Branch(key+"_q_sum_down", empty, key+"_q_sum_down"+'/I')
      for i in [1,2,3,4]:
        for j in ['iso','q_sum','tau_number']:
          nt.Branch(key+"_{}_0f_{}_up".format(j,i), empty, key+"_{}_0f_{}_up".format(j,i)+'/I')
          nt.Branch(key+"_{}_0f_{}_down".format(j,i), empty, key+"_{}_0f_{}_down".format(j,i)+'/I')
          nt.Branch(key+"_{}_1f_{}_up".format(j,i), empty, key+"_{}_0f_{}_up".format(j,i)+'/I')
          nt.Branch(key+"_{}_1f_{}_down".format(j,i), empty, key+"_{}_0f_{}_down".format(j,i)+'/I')
          nt.Branch(key+"_{}_1f1p_{}_up".format(j,i), empty, key+"_{}_1f1p_{}_up".format(j,i)+'/I')
          nt.Branch(key+"_{}_1f1p_{}_down".format(j,i), empty, key+"_{}_1f1p_{}_down".format(j,i)+'/I')
          nt.Branch(key+"_{}_1p1f_{}_up".format(j,i), empty, key+"_{}_1p1f_{}_up".format(j,i)+'/I')
          nt.Branch(key+"_{}_1p1f_{}_down".format(j,i), empty, key+"_{}_1p1f_{}_down".format(j,i)+'/I')
          nt.Branch(key+"_{}_2f_{}_up".format(j,i), empty, key+"_{}_2f_{}_up".format(j,i)+'/I')
          nt.Branch(key+"_{}_2f_{}_down".format(j,i), empty, key+"_{}_2f_{}_down".format(j,i)+'/I')
  
      nt.Branch(key+"_subtract_pass_non_closure_up", empty, key+"_subtract_pass_non_closure_up"+'/I')
      nt.Branch(key+"_subtract_pass_non_closure_down", empty, key+"_subtract_pass_non_closure_down"+'/I')
      nt.Branch(key+"_subtract_fail_non_closure_up", empty, key+"_subtract_fail_non_closure_up"+'/I')
      nt.Branch(key+"_subtract_fail_non_closure_down", empty, key+"_subtract_fail_non_closure_down"+'/I')
      nt.Branch(key+"_subtraction_up", empty, key+"_subtraction_up"+'/I')
      nt.Branch(key+"_tau_number_1_down", empty, key+"_tau_number_1_down"+'/I')
      nt.Branch(key+"_tau_number_1_up", empty, key+"_tau_number_1_up"+'/I')
      nt.Branch(key+"_tau_number_2_down", empty, key+"_tau_number_2_down"+'/I')
      nt.Branch(key+"_tau_number_2_up", empty, key+"_tau_number_2_up"+'/I')
      nt.Branch(key+"_tau_number_3_down", empty, key+"_tau_number_3_down"+'/I')
      nt.Branch(key+"_tau_number_3_up", empty, key+"_tau_number_3_up"+'/I')
      nt.Branch(key+"_tau_number_4_down", empty, key+"_tau_number_4_down"+'/I')
      nt.Branch(key+"_tau_number_4_up", empty, key+"_tau_number_4_up"+'/I')
      nt.Branch(key+"_subtraction_down", empty, key+"_subtraction_down"+'/I')

  nt.Write("ntuple",ROOT.TFile.kOverwrite)
  nf.Close()
  f.Close()
print "Finished processing"
