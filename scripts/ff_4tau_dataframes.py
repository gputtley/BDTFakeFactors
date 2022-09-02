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

parser = argparse.ArgumentParser()
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='vvvloose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
parser.add_argument('--only_mc', help= 'Only run MC dataframe',  action='store_true')
parser.add_argument('--only_data', help= 'Only run MC dataframe',  action='store_true')
args = parser.parse_args()

if args.batch:
  cmd = "python scripts/ff_4tau_dataframes.py"
  for a in vars(args):
    if "batch" in a: continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      cmd += " --{}={}".format(a,getattr(args, a))
  name = "ff_job_{}_{}_{}".format(args.channel,args.pass_wp,args.fail_wp)
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  SubmitBatchJob("jobs/"+name+".sh")
  exit()

############# Variables needed #################

years = ["2016_preVFP","2016_postVFP","2017","2018"]

fitting_variables = [
             "pt_1", "pt_2","pt_3","pt_4",
             "fabs(dphi_12)","fabs(dphi_13)","fabs(dphi_14)","fabs(dphi_23)","fabs(dphi_24)","fabs(dphi_34)",
             "fabs(dR_12)","fabs(dR_13)","fabs(dR_14)","fabs(dR_23)","fabs(dR_24)","fabs(dR_34)",
             "mt_1","mt_2","mt_3","mt_4",
             "mt_lep_12","mt_lep_13","mt_lep_14","mt_lep_23","mt_lep_24","mt_lep_34",
             "mvis_12","mvis_13","mvis_14","mvis_23","mvis_24","mvis_34",
             "pt_tt_12","pt_tt_13","pt_tt_14","pt_tt_23","pt_tt_24","pt_tt_34",
             "n_jets","n_bjets",
             ]

for ind,p in enumerate(args.channel):
  if p == "t":
    fitting_variables.append("jet_pt_"+str(ind+1))
    #fitting_variables.append("jet_probb_"+str(ind+1))
    #fitting_variables.append("jet_probbb_"+str(ind+1))
    #fitting_variables.append("jet_problepb_"+str(ind+1))
    #fitting_variables.append("jet_probc_"+str(ind+1))
    #fitting_variables.append("jet_probuds_"+str(ind+1))
    #fitting_variables.append("jet_probg_"+str(ind+1))
    fitting_variables.append("tau_decay_mode_"+str(ind+1))

############# Get dataframes #################

print "<<<< Making dataframes >>>>"

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

for t_ff in total_keys:

  print "<< Running for {} >>".format(t_ff)

  ### get selection strings ###
  sel = MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL)

  ### begin loop through different reweighting regions ###

  for rwt_key, rwt_val in sel.iteritems():

    ### get dataframes ##

    pf_df = {}

    for pf_key, pf_val in rwt_val.items():

      if not args.only_mc:
        dataset_name = "{}_{}".format(args.channel,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("deepTauVsJets","dTvJ"))
        replace = {"SELECTION":pf_val}

        if dataset_name not in datasets_created:

          print "<< Making {} {} dataframe >>".format(pf_key,rwt_key)

          pf_df = Dataframe()
          pf_df.LoadRootFilesFromJson("json_selection/ff_4tau/ff_data_{}.json".format(args.channel),fitting_variables,quiet=(args.verbosity<2),replace=replace)
          pf_df.dataframe.to_pickle("dataframes/{}_dataframe.pkl".format(dataset_name))
          if args.verbosity > 0: PrintDatasetSummary("{} {} dataframe".format(pf_key,rwt_key),pf_df.dataframe)
          datasets_created.append(dataset_name)
          del pf_df   

        else:
           print "<< Identical {} {} dataframe already made >>".format(pf_key,rwt_key)

      if not args.only_data:
        for proc in ["DY","W","ttbar"]:

          dataset_name = "{}_{}_mc_{}".format(args.channel,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("deepTauVsJets","dTvJ"),proc)

          if dataset_name not in datasets_created:

            print "<< Making {} {} {} dataframe >>".format(proc,pf_key,rwt_key)

            replace = {"SELECTION":pf_val}
            pf_df = Dataframe()
            pf_df.LoadRootFilesFromJson("json_selection/ff_4tau/ff_mc_{}.json".format(args.channel),fitting_variables,quiet=(args.verbosity<2),replace=replace,in_extra_name=proc+"_")
            pf_df.dataframe.to_pickle("dataframes/{}_dataframe.pkl".format(dataset_name))
            if args.verbosity > 0: PrintDatasetSummary("{} {} {} dataframe".format(proc,pf_key,t_ff),pf_df.dataframe)
            datasets_created.append(dataset_name)
            del pf_df

          else:

            print "<< Identical {} {} {} dataframe already made >>".format(proc,pf_key,rwt_key)
