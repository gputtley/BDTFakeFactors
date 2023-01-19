import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import itertools
import os
import json
import yaml
import re
from collections import OrderedDict
from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.batch import CreateBatchJob,SubmitBatchJob
from UserCode.BDTFakeFactors.functions import PrintDatasetSummary

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',help= 'Analysis to train BDT for', default='configs/mmtt.yaml')
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
parser.add_argument('--offset', help= 'Offset for batch jobs', type=int, default=None)
args = parser.parse_args()

############ Import config ####################

with open(args.cfg) as f: data = yaml.load(f)

############ Batch submission ################

if args.batch:
  base_cmd = "python scripts/make_dataframes.py"
  for a in vars(args):
    if "batch" in a: continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      base_cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      base_cmd += " --{}={}".format(a,getattr(args, a))

  batch_offset = 0
  for batch_offset in range(0, 18*data["channel"].count("t")):
    cmd = base_cmd + " --offset={}".format(batch_offset)
    name = "dataframe_job_{}_{}".format(data["channel"],batch_offset)
    cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
    CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
    SubmitBatchJob("jobs/"+name+".sh",cores=4)
    batch_offset += 1
  exit()

################ Functions ####################

def get_total_sel(sel,alt_nums,ind):
  ret = ""
  for num in alt_nums:
    ret += "(("+sel.replace("ALT",str(num+1))+")==1)"
    if num != alt_nums[-1]: ret += " || "
  ret.replace("NUM",str(ind+1))
  return ret

############## Make Directories ###############

if not os.path.isdir("dataframes/{}".format(data["channel"])): os.system("mkdir dataframes/{}".format(data["channel"]))
if not os.path.isdir("dataframes/{}/data_pass".format(data["channel"])): os.system("mkdir dataframes/{}/data_pass".format(data["channel"]))
if not os.path.isdir("dataframes/{}/data_fail".format(data["channel"])): os.system("mkdir dataframes/{}/data_fail".format(data["channel"]))
if not os.path.isdir("dataframes/{}/mc_fake_pass".format(data["channel"])): os.system("mkdir dataframes/{}/mc_fake_pass".format(data["channel"]))
if not os.path.isdir("dataframes/{}/mc_fake_fail".format(data["channel"])): os.system("mkdir dataframes/{}/mc_fake_fail".format(data["channel"]))
if not os.path.isdir("dataframes/{}/mc_other_pass".format(data["channel"])): os.system("mkdir dataframes/{}/mc_other_pass".format(data["channel"]))
if not os.path.isdir("dataframes/{}/mc_other_fail".format(data["channel"])): os.system("mkdir dataframes/{}/mc_other_fail".format(data["channel"]))


############### Main Loop ######################

offset = 0
for ind, obj in enumerate(data["channel"]):

  if "t" not in obj: continue

  # get numbers of other taus
  alt_tau_num = [m.start() for m in re.finditer('t', data["channel"])]
  alt_tau_num.remove(ind)

  # change variables
  variables = []
  for var in data["variables"]:
    if "NUM" in var: variables.append(var.replace("NUM",str(ind+1)))
    elif "ALT" in var:
      for alt_num in alt_tau_num: variables.append(var.replace("ALT",str(alt_num+1)))
    elif "YEAR" in var:
      continue
    else:
      variables.append(var)

  # get selections
  pass_sel = "({})".format(data["pass_selection"].replace("NUM",str(ind+1)))
  or_sideband = get_total_sel(data["sideband_pass"],alt_tau_num,ind)
  or_alt_sideband = get_total_sel(data["alt_sideband_pass"],alt_tau_num,ind)
  sels = {"mc":{},"data":{}}
  sels["data"]["pass_B"] = "(({})==1) && (({})==1) && (({})==0)".format(pass_sel, or_sideband, or_alt_sideband)
  sels["data"]["fail_B"] = "(({})==0) && (({})==1) && (({})==0)".format(pass_sel, or_sideband, or_alt_sideband)
  sels["data"]["pass_C"] = "(({})==1) && (({})==0) && (({})==1)".format(pass_sel, or_sideband, or_alt_sideband)
  sels["data"]["fail_C"] = "(({})==0) && (({})==0) && (({})==1)".format(pass_sel, or_sideband, or_alt_sideband)
  sels["data"]["pass_D"] = "(({})==1) && (({})==0) && (({})==0)".format(pass_sel, or_sideband, or_alt_sideband)
  sels["data"]["fail_D"] = "(({})==0) && (({})==0) && (({})==0)".format(pass_sel, or_sideband, or_alt_sideband)

  for k, v in sels["data"].items():
    sels["mc"]["fake_"+k] = "(({})==1) && {}".format(data["mc_gen_match"].replace("NUM",str(ind+1)),v)
    sels["mc"]["other_"+k] = "(({})==0) && {}".format(data["mc_gen_match"].replace("NUM",str(ind+1)),v)

  # set up dict to use Dataframe class - update in future
  for dm in ["data","mc"]:
    for k, v in sels[dm].items():
      if offset == args.offset or args.offset == None:
        json_file = {"add_sel":{},"file_location":data["file_location"],"lumi":1,"weights":data["mc_weight"],"baseline_sel":"(("+v+") && ("+data["baseline_selection"]+"))"}

        if dm == "data":
          json_file["data"] = 1
        else:
          json_file["data"] = 0

        if "YEAR" in data["variables"]: json_file["add_column_names"] = ["yr_"+i for i in data["years_and_lumi"].keys()]

        for yr in data["years_and_lumi"].keys():
          json_file["add_sel"][data["channel"]+"_"+yr] = {"files":data["years_and_"+dm+"_files"][yr],"sel":"(1)","file_ext":data["file_ext"].replace("YEAR",yr),"add_column_vals":[int(yr == i) for i in data["years_and_lumi"].keys()]}
          if dm == "mc": 
            json_file["add_sel"][data["channel"]+"_"+yr]["weight"] = data["years_and_lumi"][yr]
            json_file["add_sel"][data["channel"]+"_"+yr]["params_file"] = data["param_file"].replace("YEAR",yr)

        df = Dataframe()
        df.LoadRootFilesFromJson(json_file,variables,quiet=(args.verbosity<2))
        df.RenumberColumns(ind+1)
        if args.verbosity > 0: PrintDatasetSummary("{}_{}_{} dataframe".format(dm,k,str(ind+1)),df.dataframe)
        df.dataframe.to_pickle("dataframes/{}/{}_{}/{}_{}_{}.pkl".format(data["channel"], dm, k[:-2], dm, k, str(ind+1)))
      offset +=1
