from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.ff_ml import ff_ml
from UserCode.BDTFakeFactors.functions import PrintDatasetSummary, MakeSelFromList, MakeSelDict
from UserCode.BDTFakeFactors.batch import CreateBatchJob,SubmitBatchJob
from UserCode.BDTFakeFactors import reweighter
from UserCode.BDTFakeFactors.plotting import FindRebinning, RebinHist, ReplaceName, DrawTitle, DrawDistributions
from collections import OrderedDict
from array import array
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import itertools
import os
import json
import xgboost as xgb
import ROOT
import copy
ROOT.gROOT.SetBatch(ROOT.kTRUE)
ROOT.TGaxis.SetExponentOffset(-0.06, 0.01, "y");

parser = argparse.ArgumentParser()
parser.add_argument('--analysis',help= 'Analysis to train BDT for', default='4tau')
parser.add_argument('--channel',help= 'Channel to train BDT for', default='mttt')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='vvvloose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
parser.add_argument('--no_plots', help= 'Do not run plotting',  action='store_true')
parser.add_argument('--load_models', help= 'Load models in from pkl files',  action='store_true')
args = parser.parse_args()

if args.batch:
  cmd = "python scripts/ff_subtraction.py"
  for a in vars(args):
    if a == "batch": continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      cmd += " --{}={}".format(a,getattr(args, a))
  name = "ff_subtraction_job_{}_{}_{}".format(args.channel,args.pass_wp,args.fail_wp)
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  SubmitBatchJob("jobs/"+name+".sh")
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

############# Variables needed #################

logx = ["pt_1","pt_2","m_vis"]
logy = []

var_subtraction_file = open("input/{}/{}/variables/subtraction.txt".format(args.analysis,args.channel),"r")
subtraction_variables = var_subtraction_file.read().split("\n")
var_subtraction_file.close()
subtraction_variables = [s.strip() for s in subtraction_variables if s]  + ["yr_"+y for y in years]

var_plotting_file = open("input/{}/{}/variables/plotting.txt".format(args.analysis,args.channel),"r")
plotting_variables = var_plotting_file.read().split("\n")
var_plotting_file.close()
plotting_variables = [s.strip() for s in plotting_variables if s]

plot_variables = []
plot_binnings = []
for var in plotting_variables:
  if "[" in var:
    var_name = var.split("[")[0]
    bins = var.split("[")[1].split("]")[0].split(",")
    bins_float = [float(i) for i in bins]
  elif "(" in var:
    var_name = var.split("(")[0]
    bins = var.split("(")[1].split(")")[0].split(",")
    bins_float = tuple([int(bins[0]),float(bins[1]),float(bins[2])])
  plot_variables.append(var_name)
  plot_binnings.append(bins_float)

############ Functions ##########################

def ConvertDatasetName(pf_val,ch,ana):
  return "dataframes/{}/{}_{}_dataframe.pkl".format(ana,ch,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("deepTauVsJets","dTvJ"))

def ConvertMCDatasetName(pf_val,proc,ch,ana):
  return "dataframes/{}/{}_{}_mc_{}_dataframe.pkl".format(ana,ch,pf_val.replace(" ","").replace("(","").replace(")","").replace("==","_eq_").replace("!=","_neq_").replace(">=","_geq_").replace("&&","_and_").replace("deepTauVsJets","dTvJ"),proc)

############# Assign dataframes #################

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
    if (len(k)==1 and only_do_ff==k[0]) or (len(k) > 1 and multiple_ff):
      total_keys.append(k)

### Begin load and training loop ###

if not os.path.isdir("BDTs/{}".format(args.analysis)): os.system("mkdir BDTs/{}".format(args.analysis))
if not os.path.isdir("hyperparameters/{}".format(args.analysis)): os.system("mkdir hyperparameters/{}".format(args.analysis))

datasets_created = []
store = {}

for ind, t_ff in enumerate(total_keys):

  print "<< Running for {} >>".format(t_ff)

  ### Setup what dataframes you want to use and where ###
  t_ff_string = str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ","")
  sel = MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL)

  ### update variables ###
  subtraction_variables_update = []
  for var in subtraction_variables:
    if "NUM" in var:
      for i in t_ff:
        subtraction_variables_update.append(var.replace("NUM",str(i)))
    else:
      subtraction_variables_update.append(var)

  for region in ["Raw F_{F}","Alternative F_{F}","Correction"]:
 
    print "<< Doing {} >>".format(region)

    for pf in ["fail","pass"]:

      print "<< Doing {} >>".format(pf)
      model_name = "{}_{}_{}_{}_{}_{}".format(args.channel,str(args.fail_wp),args.pass_wp,region.lower().replace(" ","_").replace("_{","").replace("}",""),str(t_ff).replace("(","").replace(")","").replace(",","").replace(" ",""),pf)


      if not args.load_models:

        total_df = Dataframe()

        total_df.dataframe = pd.read_pickle(ConvertDatasetName(sel[region][pf],args.channel,args.analysis))
        total_df.dataframe = total_df.dataframe.loc[:,subtraction_variables_update+["weights"]]
        total_df.dataframe.loc[:,"y"] = 0

        for proc in mc_procs:
          mc_df = Dataframe()
          mc_df.dataframe = pd.read_pickle(ConvertMCDatasetName(sel[region][pf],proc,args.channel,args.analysis))
          mc_df.dataframe = mc_df.dataframe.loc[:,subtraction_variables_update+["weights"]]
          mc_df.dataframe.loc[:,"y"] = 1
          total_df.dataframe  = pd.concat([total_df.dataframe,mc_df.dataframe], ignore_index=True, sort=False)
          del mc_df 

        #data_sum = total_df.dataframe[(total_df.dataframe.loc[:,"y"]==0)].loc[:,"weights"].sum()
        #mc_sum = total_df.dataframe[(total_df.dataframe.loc[:,"y"]==1)].loc[:,"weights"].sum()


        ### Run training ###

        total_df.NormaliseWeightsInCategory("y", [0,1], column="weights")
        X, y, wt = total_df.SplitXyWts()

        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X, y, sample_weight=wt) 

        pkl.dump(xgb_model,open("BDTs/{}/ff_subtraction_{}.pkl".format(args.analysis,model_name), "wb"))

        #scores = xgb_model.predict_proba(total_df.dataframe[(total_df.dataframe.loc[:,"y"]==0)].loc[:,subtraction_variables_update])

        del total_df, X, y, wt

      else:

        xgb_model = pkl.load(open("BDTs/{}/ff_subtraction_{}.pkl".format(args.analysis,model_name), "rb"))


      ### finding events for removal ###

      # re get dataframes
      data_df = pd.read_pickle(ConvertDatasetName(sel[region][pf],args.channel,args.analysis))
      scoring_data_df = data_df.loc[:,subtraction_variables_update]
      data_df.loc[:,"scores"] = xgb_model.predict_proba(scoring_data_df)[:,1]

      for proc in mc_procs:
        mc_df_temp = pd.read_pickle(ConvertMCDatasetName(sel[region][pf],proc,args.channel,args.analysis))
        if proc == mc_procs[0]:
          mc_df = copy.deepcopy(mc_df_temp)
        else:
          mc_df  = pd.concat([mc_df,mc_df_temp], ignore_index=True, sort=False)
      scoring_mc_df = mc_df.loc[:,subtraction_variables_update] 
      mc_df.loc[:,"scores"] = xgb_model.predict_proba(scoring_mc_df)[:,1]

      nbins = 1000
      mc_hist, mc_bins = np.histogram(np.array(mc_df.loc[:,"scores"]),bins=nbins,weights=np.array(mc_df.loc[:,"weights"]),range=(0,1))

      rm_indices = []
      for ind, val in enumerate(mc_hist):
        # get slimmed down data df
        if mc_bins[ind+1] == 1.0: mc_bins[ind+1] = 1.00001
        slimmed_data_df = data_df[((data_df.loc[:,"scores"]>=mc_bins[ind]) & (data_df.loc[:,"scores"]<mc_bins[ind+1]))]

        # remove events
        if len(slimmed_data_df) > 0:
          removed_weights = 0.0
          while removed_weights < val:
            sample = slimmed_data_df.sample()
            rm_indices.append(sample.index.values[0])
            removed_weights += sample.loc[:,"weights"].iloc[0]

      ### Remove events ###
      indexes_to_keep = set(range(data_df.shape[0])) - set(rm_indices)

      subtract_df = Dataframe()
      subtract_df.dataframe = data_df.take(list(indexes_to_keep))

      subtract_df.dataframe.drop(["scores"],axis=1,inplace=True)

      ### write dataframes ###
      subtract_df.dataframe.to_pickle(ConvertDatasetName(sel[region][pf],args.channel,args.analysis).replace(".pkl","_subtraction.pkl"))

      ### plotting ###
      subtract_df.dataframe = subtract_df.dataframe.loc[:,plot_variables+["weights"]] 
      ttree_subtract = subtract_df.WriteToRoot(None,key='ntuple',return_tree=True)
      del subtract_df

      # get data df + mc df with negative weights
      neg_weight_df = Dataframe()
      neg_weight_df.dataframe = pd.read_pickle(ConvertDatasetName(sel[region][pf],args.channel,args.analysis))

      no_subtract_df = Dataframe()
      no_subtract_df.dataframe = neg_weight_df.dataframe.loc[:,plot_variables+["weights"]]
      ttree_nom = no_subtract_df.WriteToRoot(None,key='ntuple',return_tree=True)
      del no_subtract_df


      for proc in mc_procs:
        mc_df = Dataframe()
        mc_df.dataframe = pd.read_pickle(ConvertMCDatasetName(sel[region][pf],proc,args.channel,args.analysis))
        mc_df.dataframe.loc[:,"weights"] = -1 * mc_df.dataframe.loc[:,"weights"]
        neg_weight_df.dataframe  = pd.concat([neg_weight_df.dataframe,mc_df.dataframe], ignore_index=True, sort=False)

      neg_weight_df.dataframe = neg_weight_df.dataframe.loc[:,plot_variables+["weights"]]     
      ttree_neg_weight = neg_weight_df.WriteToRoot(None,key='ntuple',return_tree=True)

      # loop through variables
      if not args.no_plots:
        for var_ind, var_name in enumerate(plot_variables):

          var_binning = plot_binnings[var_ind]
          if isinstance(var_binning,tuple):
            bins = array('f', map(float,[(float(var_binning[2]-var_binning[1])/var_binning[0])*i + float(var_binning[1]) for i in range(0,var_binning[0])]))
          elif isinstance(var_binning,list):
            bins = array('f', map(float,var_binning))

          c_clos = ROOT.TCanvas('c','c',600,600)

          hout = ROOT.TH1D('hout','',len(bins)-1, bins)
          for i in range(0,hout.GetNbinsX()+2): hout.SetBinError(i,0)
          hist1 = hout.Clone()
          hist1.SetName('hist1')
          hist2 = hout.Clone()
          hist2.SetName('hist2')
          hist3 = hout.Clone()
          hist3.SetName('hist3')

          hist1.SetDirectory(ROOT.gROOT)
          ttree_subtract.Draw('%(var_name)s>>hist1' % vars(),'weights*(1)','goff')
          hist1 = ttree_subtract.GetHistogram()

          hist2.SetDirectory(ROOT.gROOT)
          ttree_neg_weight.Draw('%(var_name)s>>hist2' % vars(),'weights*(1)','goff')
          hist2 = ttree_neg_weight.GetHistogram()

          hist3.SetDirectory(ROOT.gROOT)
          ttree_nom.Draw('%(var_name)s>>hist3' % vars(),'weights*(1)','goff')
          hist3 = ttree_nom.GetHistogram()

          if neg_weight_df.dataframe.loc[:,var_name].nunique() > 20:
            BinThreshold=100
            BinUncertFraction=0.1
            binning = FindRebinning(hist1,BinThreshold=BinThreshold,BinUncertFraction=BinUncertFraction)
            hist2 = RebinHist(hist2,binning)
            binning = FindRebinning(hist2,BinThreshold=BinThreshold,BinUncertFraction=BinUncertFraction)
            hist1 = RebinHist(hist1,binning)
            hist2 = RebinHist(hist2,binning)
            hist3 = RebinHist(hist3,binning)
        
          hist1_divide = hist1.Clone()
          for i in range(0,hist1_divide.GetNbinsX()+2): hist1_divide.SetBinError(i,0)

          hist1_hist1_ratio = hist1.Clone()
          hist1_hist1_ratio.Divide(hist1_divide)

          hist2_hist1_ratio = hist2.Clone()
          hist2_hist1_ratio.Divide(hist1_divide)

          hist3_hist1_ratio = hist3.Clone()
          hist3_hist1_ratio.Divide(hist1_divide)

          c_clos = ROOT.TCanvas('c_clos','c_clos',600,600)

          pad1 = ROOT.TPad("pad1","pad1",0,0.37,1,1)
          pad1.SetBottomMargin(0.02)
          pad1.SetLeftMargin(0.15)
          if var_name in logx: pad1.SetLogx()
          if var_name in logy: pad1.SetLogy()
          pad1.Draw()
          pad1.cd()

          hist1.Draw("hist SAME")
          hist1.SetLineColor(1)
          hist1.SetFillColor(38)
          hist1.SetStats(0)
          hist1.GetXaxis().SetTitle("")
          hist1.GetYaxis().SetTitle(ReplaceName("dN/d{}".format(var_name)))
          hist1.GetYaxis().SetTitleOffset(1.2)
          hist1.GetYaxis().SetTitleSize(0.06)
          hist1.GetYaxis().SetLabelSize(0.05)
          hist1.GetXaxis().SetLabelSize(0)
          hist1.SetTitle("")

          hist1_uncert = hist2.Clone()
          hist1_uncert.SetMarkerSize(0)
          hist1_uncert.SetFillColorAlpha(12,0.5)
          hist1_uncert.SetLineWidth(0)
          hist1_uncert.Draw("e2 same")

          hist2.Draw("E SAME")
          hist2.SetMarkerColor(1)
          hist2.SetLineColor(1)
          hist2.SetMarkerStyle(19)

          hist3.Draw("E SAME")
          hist3.SetMarkerColor(2)
          hist3.SetLineColor(2)
          hist3.SetMarkerStyle(2)

          l = ROOT.TLegend(0.6,0.65,0.88,0.85);
          l.SetBorderSize(0)
          l.AddEntry(hist1,"BDT subtraction","f")
          l.AddEntry(hist2,"Histogram subtraction","lep")
          l.AddEntry(hist3,"No subtraction","lep")
          l.Draw()

          c_clos.cd()
          pad2 = ROOT.TPad("pad2","pad2",0,0.05,1,0.35)
          pad2.SetLeftMargin(0.15)
          pad2.SetTopMargin(0.03)
          pad2.SetBottomMargin(0.3)
          if var_name in logx: pad2.SetLogx()
          pad2.Draw()
          pad2.cd()

          ratio_line = ROOT.TLine(hist1.GetBinLowEdge(1),1,hist1.GetBinLowEdge(hist1.GetNbinsX()+1),1)
          hist1_hist1_ratio.SetTitle("")
          hist1_hist1_ratio.SetMarkerSize(0)
          hist1_hist1_ratio.SetFillColorAlpha(12,0.5)
          hist1_hist1_ratio.SetLineWidth(0)
          hist1_hist1_ratio.SetAxisRange(0.6,1.4,'Y')
          hist1_hist1_ratio.GetYaxis().SetNdivisions(4)
          hist1_hist1_ratio.SetStats(0)
          hist1_hist1_ratio.GetXaxis().SetLabelSize(0.1)
          hist1_hist1_ratio.GetYaxis().SetLabelSize(0.1)
          hist1_hist1_ratio.GetXaxis().SetTitle(ReplaceName(var_name))
          hist1_hist1_ratio.GetYaxis().SetTitle("Ratio")
          hist1_hist1_ratio.GetYaxis().SetTitleColor(1)
          hist1_hist1_ratio.GetYaxis().SetTitleSize(0.12)
          hist1_hist1_ratio.GetYaxis().SetTitleOffset(0.5)
          hist1_hist1_ratio.GetXaxis().SetTitleSize(0.12)
          hist1_hist1_ratio.GetXaxis().SetTitleOffset(1.2)
          if var_name in logx:
            hist1_hist1_ratio.GetXaxis().SetMoreLogLabels()
            hist1_hist1_ratio.GetXaxis().SetNoExponent()

          hist2_hist1_ratio.SetMarkerColor(1)
          hist2_hist1_ratio.SetLineColor(1)
          hist2_hist1_ratio.SetMarkerStyle(19)

          hist3_hist1_ratio.SetMarkerColor(2)
          hist3_hist1_ratio.SetLineColor(2)
          hist3_hist1_ratio.SetMarkerStyle(2)

          ratio_line_up = ROOT.TLine(hist1.GetBinLowEdge(1),1.2,hist1.GetBinLowEdge(hist1.GetNbinsX()+1),1.2)
          ratio_line_down = ROOT.TLine(hist1.GetBinLowEdge(1),0.8,hist1.GetBinLowEdge(hist1.GetNbinsX()+1),0.8)
          ratio_line.SetLineStyle(3)
          ratio_line_up.SetLineStyle(3)
          ratio_line_down.SetLineStyle(3)

          hist1_hist1_ratio.Draw("e2")
          ratio_line.Draw("l same")
          ratio_line_up.Draw("l same")
          ratio_line_down.Draw("l same")
          hist2_hist1_ratio.Draw("E same")
          hist3_hist1_ratio.Draw("E same")

          title_left=args.channel
          title_right="all_years"
          DrawTitle(pad1, ReplaceName(title_left), 1, scale=1)
          DrawTitle(pad1, ReplaceName(title_right), 3, scale=1)

          c_clos.Update()
          rwt_key_name = region.lower().replace(" ","_").replace("_{","").replace("}","")
          plot_save_name = "_{}_{}_{}".format(t_ff_string,rwt_key_name,pf,args.channel)
          clos_name = 'plots/subtraction_plot_%(var_name)s%(plot_save_name)s.pdf' % vars()
          c_clos.SaveAs(clos_name)
          c_clos.Close()
          del c_clos

