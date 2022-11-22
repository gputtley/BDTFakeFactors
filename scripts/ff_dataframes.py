from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.functions import PrintDatasetSummary, MakeSelFromList, MakeSelDict
from UserCode.BDTFakeFactors.batch import CreateBatchJob,SubmitBatchJob
from collections import OrderedDict
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import itertools
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--analysis',help= 'Analysis to train BDT for', default='4tau')
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='vvvloose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
parser.add_argument('--only_mc', help= 'Only run MC dataframe',  action='store_true')
parser.add_argument('--only_data', help= 'Only run data dataframe',  action='store_true')
parser.add_argument('--remove_weights', help= 'Remove weights from dataframe',  action='store_true')
parser.add_argument('--offset', help= 'Offset for batch jobs', type=int, default=None)
args = parser.parse_args()

############ Import config ####################

with open("input/{}/{}/config.json".format(args.analysis,args.channel)) as jsonfile:
  config = json.load(jsonfile)

years = config["years"]
SIDEBAND = config["DR"]
SIGNAL = config["SR"]
mc_procs = config["mc_procs"]
multiple_ff = config["multiple_ff"]
do_subtraction = config["do_subtraction"]

############# Variables needed #################

var_fitting_file = open("input/{}/{}/variables/fitting.txt".format(args.analysis,args.channel),"r")
fitting_variables = var_fitting_file.read().split("\n")
var_fitting_file.close()
fitting_variables = [s.strip() for s in fitting_variables if s]

var_scoring_file = open("input/{}/{}/variables/scoring.txt".format(args.analysis,args.channel),"r")
scoring_variables = var_scoring_file.read().split("\n")
var_scoring_file.close()
scoring_variables = [s.strip() for s in scoring_variables if s]

if do_subtraction:
  var_subtraction_file = open("input/{}/{}/variables/subtraction.txt".format(args.analysis,args.channel),"r")
  subtraction_variables = var_subtraction_file.read().split("\n")
  var_subtraction_file.close()
  subtraction_variables = [s.strip() for s in subtraction_variables if s]
else:
  subtraction_variables = []

if os.path.exists("input/{}/{}/variables/other.txt".format(args.analysis,args.channel)):
  var_other_file = open("input/{}/{}/variables/other.txt".format(args.analysis,args.channel),"r")
  other_variables = var_other_file.read().split("\n")
  var_other_file.close()
  other_variables = [s.strip() for s in other_variables if s]
else:
  other_variables = []

var_plotting_file = open("input/{}/{}/variables/plotting.txt".format(args.analysis,args.channel),"r")
plotting_variables = var_plotting_file.read().split("\n")
var_plotting_file.close()
plotting_variables = [s.strip() for s in plotting_variables if s]

total_variables = list(set(fitting_variables + scoring_variables + subtraction_variables + other_variables))

for var in plotting_variables:
  if "[" in var:
    var_name = var.split("[")[0]
  elif "(" in var:
    var_name = var.split("(")[0]
  if var_name not in total_variables: total_variables.append(var_name)

variables = []
for var in total_variables:
  if "NUM" in var or "ALT" in var:
    for ind,i in enumerate(args.channel):
      if i == "t":
        variables.append(var.replace("NUM",str(ind+1)).replace("ALT",str(ind+1)))
  else:
    variables.append(var)

variables = list(set(variables))


############# Get dataframes #################

print "<<<< Making dataframes >>>>"

### setup pass and fail ###

if args.fail_wp == None:
  FAIL = "(deepTauVsJets_iso_X>=0 && deepTauVsJets_{}_X==0)".format(args.pass_wp)
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
    if len(k)==1 or (len(k) > 1 and multiple_ff):
      total_keys.append(k)


### setup batch jobs ##

if args.batch:
  base_cmd = "python scripts/ff_dataframes.py"
  for a in vars(args):
    if "batch" in a: continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      base_cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      base_cmd += " --{}={}".format(a,getattr(args, a))

  batch_offset = 0
  for t_ff in total_keys:
    sel = MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL)
    for rwt_key, rwt_val in sel.iteritems():
      cmd = base_cmd + " --offset={}".format(batch_offset)
      name = "ff_dataframe_job_{}_{}_{}_{}".format(args.channel,args.pass_wp,args.fail_wp,batch_offset)
      cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
      CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
      SubmitBatchJob("jobs/"+name+".sh")
      batch_offset += 1
  exit()

### Begin load loop ###

if not os.path.isdir("dataframes/{}".format(args.analysis)): os.system("mkdir dataframes/{}".format(args.analysis))

datasets_created = []

offset = 0
for t_ff in total_keys:

  print "<< Running for {} >>".format(t_ff)

  ### get selection strings ###
  sel = MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL)

  ### begin loop through different reweighting regions ###

  for rwt_key, rwt_val in sel.iteritems():

    ### get dataframes ##

    pf_df = {}

    for pf_key, pf_val in rwt_val.items():

      if (not args.only_mc) and (args.offset == offset or args.offset == None):
        dataset_name = "{}_{}".format(args.channel,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("deepTauVsJets","dTvJ"))
        replace = {"SELECTION":pf_val}

        if dataset_name not in datasets_created:

          print "<< Making {} {} dataframe >>".format(pf_key,rwt_key)

          pf_df = Dataframe()
          pf_df.LoadRootFilesFromJson("input/{}/{}/selections/data.json".format(args.analysis,args.channel),variables,quiet=(args.verbosity<2),replace=replace)
          if args.remove_weights: pf_df.dataframe = pf_df.dataframe.drop(["weights"],axis=1)
          pf_df.dataframe.to_pickle("dataframes/{}/{}_dataframe.pkl".format(args.analysis,dataset_name))
          if args.verbosity > 0: PrintDatasetSummary("{} {} dataframe".format(pf_key,rwt_key),pf_df.dataframe)
          datasets_created.append(dataset_name)
          del pf_df

        else:
           print "<< Identical {} {} dataframe already made >>".format(pf_key,rwt_key)

      if (not args.only_data) and (args.offset == offset or args.offset == None):
        for proc in mc_procs:

          dataset_name = "{}_{}_mc_{}".format(args.channel,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("deepTauVsJets","dTvJ"),proc)

          if dataset_name not in datasets_created:

            print "<< Making {} {} {} dataframe >>".format(proc,pf_key,rwt_key)

            replace = {"SELECTION":pf_val}
            pf_df = Dataframe()
            pf_df.LoadRootFilesFromJson("input/{}/{}/selections/mc.json".format(args.analysis,args.channel),variables,quiet=(args.verbosity<2),replace=replace,in_extra_name=proc+"_")
            if args.remove_weights: pf_df.dataframe = pf_df.dataframe.drop(["weights"],axis=1)
            pf_df.dataframe.to_pickle("dataframes/{}/{}_dataframe.pkl".format(args.analysis,dataset_name))
            if args.verbosity > 0: PrintDatasetSummary("{} {} {} dataframe".format(proc,pf_key,t_ff),pf_df.dataframe)
            datasets_created.append(dataset_name)
            del pf_df

          else:

            print "<< Identical {} {} {} dataframe already made >>".format(proc,pf_key,rwt_key)

      offset += 1
