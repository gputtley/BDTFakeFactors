from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.functions import PrintDatasetSummary, GetNonZeroMinimum
from UserCode.BDTFakeFactors.batch import CreateBatchJob,SubmitBatchJob
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
import yaml
import random
from sklearn import metrics
ROOT.gROOT.SetBatch(ROOT.kTRUE)
ROOT.TGaxis.SetExponentOffset(-0.06, 0.01, "y");

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',help= 'Config to pass', default='configs/mmtt.yaml')
parser.add_argument('--batch', help= 'Run job on the batch',  action='store_true')
parser.add_argument('--verbosity', help= 'Changes how much is printed', type=int, default=0)
parser.add_argument('--no_plots', help= 'Do not run plotting',  action='store_true')
parser.add_argument('--load_models', help= 'Load models in from pkl files',  action='store_true')
args = parser.parse_args()

############ Import config ####################

with open(args.cfg) as f: data = yaml.load(f)

############ Batch Submission ####################

if args.batch:
  cmd = "python scripts/do_subtraction.py"
  for a in vars(args):
    if a == "batch": continue
    if getattr(args, a) == True and type(getattr(args, a)) == bool:
      cmd += " --{}".format(a)
    elif getattr(args, a) != False and getattr(args, a) != None:
      cmd += " --{}={}".format(a,getattr(args, a))
  name = "subtraction_job_{}".format(data["channel"])
  cmssw_base = os.getcwd().replace('src/UserCode/BDTFakeFactors','')
  CreateBatchJob("jobs/"+name+".sh",cmssw_base,[cmd])
  SubmitBatchJob("jobs/"+name+".sh")
  exit()

############ Functions ##########################


def FindBinning(df,col,n_bins=20,ignore_quantile=0.01):
  if len(df.loc[:,col].unique()) > n_bins: # continuous
    bins = list(np.linspace(df.loc[:,col].quantile(ignore_quantile),df.loc[:,col].quantile(1-ignore_quantile),n_bins))
#    bins[0] = df.loc[:,col].min()
#    bins[-1] = df.loc[:,col].max()
  else:
    bins = list(np.linspace(df.loc[:,col].min(),df.loc[:,col].max()+1,int(df.loc[:,col].max()-df.loc[:,col].min())+2))
  bins = array('f', map(float,bins))
  return bins

############# Assign dataframes #################

### Begin load and training loop ###

if not os.path.isdir("BDTs/{}".format(data["channel"])): os.system("mkdir BDTs/{}".format(data["channel"]))
if not os.path.isdir("hyperparameters/{}".format(data["channel"])): os.system("mkdir hyperparameters/{}".format(data["channel"]))
if not os.path.isdir("dataframes/{}/subtracted_pass".format(data["channel"])): os.system("mkdir dataframes/{}/subtracted_pass".format(data["channel"]))
if not os.path.isdir("dataframes/{}/subtracted_fail".format(data["channel"])): os.system("mkdir dataframes/{}/subtracted_fail".format(data["channel"]))

for pf in ["pass","fail"]:
#for pf in ["pass"]:

  if not args.load_models:

    print "<< Doing {} >>".format(pf)

    print "<< Loading Dataframes >>"
    ind = 0
    for ft in ["fake","other"]:
      fdir = "dataframes/{}/mc_{}_{}".format(data["channel"],ft,pf)
      for val in os.listdir(fdir):
        temp_df = pd.read_pickle(fdir+"/"+val)
        temp_df.loc[:,"y"] = int(ft=="fake")
        if ind == 0:
          mc_df = Dataframe()
          mc_df.dataframe = copy.deepcopy(temp_df)
        else:
          mc_df.dataframe = pd.concat([temp_df,mc_df.dataframe], ignore_index=True, sort=False)
        ind += 1
          
    print "<< Running Training >>"

    mc_df.NormaliseWeightsInCategory("y", [0,1], column="weights")
    X, y, wt = mc_df.SplitXyWts()

    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X, y, sample_weight=wt) 

    scores = xgb_model.predict_proba(X)[:,1]
    print metrics.roc_auc_score(y,scores)

    pkl.dump(xgb_model,open("BDTs/{}/subtraction_{}.pkl".format(data["channel"],pf), "wb"))
    del temp_df, X, y, wt

  else:

    print "<< Loading Model >>"

    xgb_model = pkl.load(open("BDTs/{}/subtraction_{}.pkl".format(data["channel"],pf), "rb"))


  ### finding events for removal ###

  dfdir = "dataframes/{}/data_{}".format(data["channel"],pf)
  for filename in os.listdir(dfdir):
    data_df = pd.read_pickle(dfdir+"/"+filename)
    X_data_df = copy.deepcopy(data_df)
    X_data_df.drop(["weights"],axis=1,inplace=True)
    data_df.loc[:,"scores"] = xgb_model.predict_proba(X_data_df)[:,1]

    mcdfdir = "dataframes/{}/mc_other_{}/mc_other_{}".format(data["channel"],pf,filename.replace("data_",""))      
    mc_df = pd.read_pickle(mcdfdir)
    X_mc_df = copy.deepcopy(mc_df)
    X_mc_df.drop(["weights"],axis=1,inplace=True)
    mc_df.loc[:,"scores"] = xgb_model.predict_proba(X_mc_df)[:,1]

    nbins = max(min(int(np.floor(mc_df.loc[:,"weights"].sum()/2)),100),5)
    mc_hist, mc_bins = np.histogram(np.array(mc_df.loc[:,"scores"]),bins=nbins,weights=np.array(mc_df.loc[:,"weights"]),range=(0,1))

    rm_indices = []
    for ind, val in enumerate(mc_hist):
      # get slimmed down data df
      if mc_bins[ind+1] == 1.0: mc_bins[ind+1] = 1.00001
      slimmed_data_df = data_df[((data_df.loc[:,"scores"]>=mc_bins[ind]) & (data_df.loc[:,"scores"]<mc_bins[ind+1]))]

      # remove events
      if len(slimmed_data_df) > 0:
        removed_weights = 0.0
        sample = slimmed_data_df.sample()
        stuck_counter = 0
        # this removes weights that are under, but half the time they will be over. We could do some bin merging if we want to be clever
        while removed_weights+sample.loc[:,"weights"].iloc[0] < val or (removed_weights < val and removed_weights+sample.loc[:,"weights"].iloc[0] > val and val > 0.5 and bool(random.getrandbits(1))):
          if sample.index.values[0] not in rm_indices:
            rm_indices.append(sample.index.values[0])
            removed_weights += sample.loc[:,"weights"].iloc[0]
            stuck_counter = 0
          elif stuck_counter > 5:
            break
          else:
            stuck_counter += 1 
          sample = slimmed_data_df.sample()
 
    ### Remove events ###
    indexes_to_keep = set(range(data_df.shape[0])) - set(rm_indices)
    rm_indices.sort()
    subtract_df = Dataframe()
    subtract_df.dataframe = data_df.take(list(indexes_to_keep))
    subtract_df.dataframe.drop(["scores"],axis=1,inplace=True)
    print "Subtracted Sum", subtract_df.dataframe.loc[:,"weights"].sum()
    print "Data Sum - MC Sum", data_df.loc[:,"weights"].sum() - mc_df.loc[:,"weights"].sum()
    ### write dataframes ###
    subtract_df.dataframe.to_pickle("dataframes/{}/subtracted_{}/subtracted_{}".format(data["channel"],pf,filename.replace("data_","")))

  ### Plotting ###

  ind = 0
  fdir = "dataframes/{}/subtracted_{}".format(data["channel"],pf)
  for val in os.listdir(fdir):
    temp_df = pd.read_pickle(fdir+"/"+val)
    if ind == 0:
      subtract_df = Dataframe()
      subtract_df.dataframe = copy.deepcopy(temp_df)
    else:
      subtract_df.dataframe = pd.concat([temp_df,subtract_df.dataframe], ignore_index=True, sort=False)
    ind += 1

  ind = 0 
  fdir = "dataframes/{}/mc_other_{}".format(data["channel"],pf)
  for val in os.listdir(fdir):
    temp_df = pd.read_pickle(fdir+"/"+val)
    if ind == 0:
      mc_df = Dataframe()
      mc_df.dataframe = copy.deepcopy(temp_df)
    else:
      mc_df.dataframe = pd.concat([temp_df,mc_df.dataframe], ignore_index=True, sort=False)
    ind += 1 

  ind = 0 
  fdir = "dataframes/{}/data_{}".format(data["channel"],pf)
  for val in os.listdir(fdir):
    temp_df = pd.read_pickle(fdir+"/"+val)
    if ind == 0:
      neg_weight_df = Dataframe()
      neg_weight_df.dataframe = copy.deepcopy(temp_df)
    else:
      neg_weight_df.dataframe = pd.concat([temp_df,neg_weight_df.dataframe], ignore_index=True, sort=False)
    ind += 1 

  ttree_subtract = subtract_df.WriteToRoot(None,key='ntuple',return_tree=True)
  ttree_nom = copy.deepcopy(neg_weight_df).WriteToRoot(None,key='ntuple',return_tree=True)
  mc_df.dataframe.loc[:,"weights"] = -1 * mc_df.dataframe.loc[:,"weights"]
  neg_weight_df.dataframe  = pd.concat([neg_weight_df.dataframe,mc_df.dataframe], ignore_index=True, sort=False)
  ttree_neg_weight = neg_weight_df.WriteToRoot(None,key='ntuple',return_tree=True)


  # loop through variables
  if not args.no_plots:
    for var_ind, var_name in enumerate(subtract_df.dataframe.columns):
      if var_name == "weights": continue

      bins = FindBinning(subtract_df.dataframe,var_name,n_bins=20,ignore_quantile=0.01)
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

      if subtract_df.dataframe.loc[:,var_name].nunique() > 10:
        BinThreshold=100
        BinUncertFraction=0.1
        binning = FindRebinning(hist1,BinThreshold=BinThreshold,BinUncertFraction=BinUncertFraction)
        hist2 = RebinHist(hist2,binning)
        binning = FindRebinning(hist2,BinThreshold=BinThreshold,BinUncertFraction=BinUncertFraction)
        hist1 = RebinHist(hist1,binning)
        hist2 = RebinHist(hist2,binning)
        hist3 = RebinHist(hist3,binning)
   
      # norm to bin width
      hist1.Scale(1.0,"width")
      hist2.Scale(1.0,"width")
      hist3.Scale(1.0,"width")

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
      #if var_name in logx: pad1.SetLogx()
      #if var_name in logy: pad1.SetLogy()
      pad1.Draw()
      pad1.cd()

      hist1.Draw("hist SAME")
      hist1.SetLineColor(1)
      hist1.SetFillColor(38)
      hist1.SetStats(0)
      hist1.SetMinimum(0)
      hist1.SetMaximum(1.5*hist1.GetMaximum())
      hist1.GetXaxis().SetTitle("")
      hist1.GetYaxis().SetTitle(ReplaceName("dN/d{}".format(var_name)))
      hist1.GetYaxis().SetTitleOffset(1.2)
      hist1.GetYaxis().SetTitleSize(0.06)
      hist1.GetYaxis().SetLabelSize(0.05)
      hist1.GetXaxis().SetLabelSize(0)
      hist1.SetTitle("")

      hist1_uncert = hist1.Clone()
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
      #if var_name in logx: pad2.SetLogx()
      pad2.Draw()
      pad2.cd()

      max_ratio = max(hist3_hist1_ratio.GetMaximum(),hist2_hist1_ratio.GetMaximum())
      min_ratio = min(hist3_hist1_ratio.GetMinimum(),hist2_hist1_ratio.GetMinimum())
      possible_max = [1.05,1.1,1.2,1.4,1.6,1.8,2.0,3.0,4.0,5.0,10.0]
      possible_min = [0.95,0.9,0.8,0.6,0.4,0.2,0.0]
      max_ratio = min(possible_max, key=lambda x:abs(x-max_ratio))
      min_ratio = min(possible_min, key=lambda x:abs(x-min_ratio))
      if min_ratio == 0.0: min_ratio = max(2 - max_ratio,0)

      orig_max_ratio = max([i.GetMaximum() for i in [hist3_hist1_ratio,hist2_hist1_ratio]])
      orig_min_ratio = min([GetNonZeroMinimum(i) for i in [hist3_hist1_ratio,hist2_hist1_ratio]])
      largest_shift = max(orig_max_ratio,2-orig_min_ratio)
      possible_max = [1.05,1.1,1.2,1.4,1.6,1.8,2.0,3.0,4.0,5.0,10.0]
      max_ratio = min(possible_max, key=lambda x:((x>largest_shift)*abs(x-largest_shift)) + ((x<=largest_shift)*9999))
      min_ratio = max(2 - max_ratio,0)


      ratio_line = ROOT.TLine(hist1.GetBinLowEdge(1),1,hist1.GetBinLowEdge(hist1.GetNbinsX()+1),1)
      hist1_hist1_ratio.SetTitle("")
      hist1_hist1_ratio.SetMarkerSize(0)
      hist1_hist1_ratio.SetFillColorAlpha(12,0.5)
      hist1_hist1_ratio.SetLineWidth(0)
      hist1_hist1_ratio.SetAxisRange(min_ratio,max_ratio,'Y')
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
      #if var_name in logx:
      #  hist1_hist1_ratio.GetXaxis().SetMoreLogLabels()
      #  hist1_hist1_ratio.GetXaxis().SetNoExponent()

      hist2_hist1_ratio.SetMarkerColor(1)
      hist2_hist1_ratio.SetLineColor(1)
      hist2_hist1_ratio.SetMarkerStyle(19)

      hist3_hist1_ratio.SetMarkerColor(2)
      hist3_hist1_ratio.SetLineColor(2)
      hist3_hist1_ratio.SetMarkerStyle(2)

      ratio_line_up = ROOT.TLine(hist1.GetBinLowEdge(1),(1+max_ratio)/2,hist1.GetBinLowEdge(hist1.GetNbinsX()+1),(1+max_ratio)/2)
      ratio_line_down = ROOT.TLine(hist1.GetBinLowEdge(1),(1+min_ratio)/2,hist1.GetBinLowEdge(hist1.GetNbinsX()+1),(1+min_ratio)/2)
      ratio_line.SetLineStyle(3)
      ratio_line_up.SetLineStyle(3)
      ratio_line_down.SetLineStyle(3)

      hist1_hist1_ratio.Draw("e2")
      ratio_line.Draw("l same")
      ratio_line_up.Draw("l same")
      ratio_line_down.Draw("l same")
      hist2_hist1_ratio.Draw("E same")
      hist3_hist1_ratio.Draw("E same")

      title_left=data["channel"]
      title_right="all_years"
      if pf == "pass":
        DrawTitle(pad1, ReplaceName(title_left)+" Pass Region", 1, scale=1)
      elif pf == "fail":
        DrawTitle(pad1, ReplaceName(title_left)+" Fail Region", 1, scale=1)
      DrawTitle(pad1, ReplaceName(title_right), 3, scale=1)

      c_clos.Update()
      plot_save_name = "_{}_{}".format(data["channel"],pf)
      clos_name = 'plots/subtraction_plot_%(var_name)s%(plot_save_name)s.pdf' % vars()
      print "Created", clos_name
      c_clos.SaveAs(clos_name)
      c_clos.Close()
      del c_clos
