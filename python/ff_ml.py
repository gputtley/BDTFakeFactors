from UserCode.BDTFakeFactors import reweighter
from UserCode.BDTFakeFactors.Dataframe import Dataframe
#from UserCode.BDTFakeFactors.plotting import DrawClosurePlots, DrawColzReweightPlots, DrawAverageReweightPlots, FindRebinning, RebinHist, ReplaceName
from UserCode.BDTFakeFactors.plotting import FindRebinning, RebinHist, ReplaceName, DrawTitle
from UserCode.BDTFakeFactors.functions import GetNonZeroMinimum, GetMaximum, GetNonClosureLargestShift
import copy
import pandas as pd
import numpy as np
import ROOT
import gc
from UserCode.BDTFakeFactors import reweighter
from array import array
from hep_ml.metrics_utils import ks_2samp_weighted, compute_cdf

ROOT.gErrorIgnoreLevel = ROOT.kFatal

class ff_ml():

  def __init__(self):
    self.df1 = {"C":[], "D":[], "B":[], "Signal":[], "All":[]}
    self.df2 = {"C":[], "D":[], "B":[], "Signal":[], "All":[]}
    self.model = {"C":None, "D":None, "B":None, "Signal":None, "All":None}

  def AddDataframes(self,df1, df2, name):
    if name not in ["C","D","B","Signal","All"]:
      print "ERROR: Invalid name given. Please give Raw, Alternative, B or Signal"
    else:
      if df1 != None:
        self.df1[name].append(copy.deepcopy(df1))
        if not name == "Signal": self.df1["All"].append(copy.deepcopy(df1))
      if df2 != None:
        self.df2[name].append(copy.deepcopy(df2))
        if not name == "Signal": self.df2["All"].append(copy.deepcopy(df2))

  def AddModel(self,model,name):
    if name not in ["C","D","B","Signal","All"]:
      print "ERROR: Invalid name given. Please give Raw, Alternative, B or Signal"
    else:
      self.model[name] = copy.deepcopy(model)
      if name == "All":
        self.model["C"] = model
        self.model["D"] = model
        self.model["B"] = model

  def GetDataframes(self,name,data_type=None,with_reweights=False,full=False,variables="all",all_mode=False,return_specific=None):
    if name not in ["C","D","B","Signal","All"]:
      print "ERROR: Invalid name given. Please give Raw, Alternative, B or Signal"
    else:

      if return_specific == None or return_specific == 1:
        df1 = Dataframe()
        for ind1,i1 in enumerate(self.df1[name]):
          if ind1 == 0:
            df1.dataframe = pd.read_pickle(i1)
          else:
            df1.dataframe  = pd.concat([df1.dataframe,pd.read_pickle(i1)], ignore_index=True, sort=False)


      if return_specific == None or return_specific == 2:
        df2 = Dataframe()
        for ind2,i2 in enumerate(self.df2[name]):
          if ind2 == 0:
            df2.dataframe = pd.read_pickle(i2)
          else:
            df2.dataframe  = pd.concat([df2.dataframe,pd.read_pickle(i2)], ignore_index=True, sort=False)

    if variables != "all":
      if "weights" not in variables: variables.append("weights")
      if return_specific == None or return_specific == 1: df1.dataframe = df1.dataframe.loc[:,variables]
      if return_specific == None or return_specific == 2: df2.dataframe = df2.dataframe.loc[:,variables]

    if data_type == "train":
      if return_specific == None or return_specific == 1:
        df1.TrainTestSplit()
        X_df1, wt_df1, _, _ = df1.SplitTrainTestXWts()
      if return_specific == None or return_specific == 2:
        df2.TrainTestSplit()
        X_df2, wt_df2, _, _ = df2.SplitTrainTestXWts()
    elif data_type == "test":
      if return_specific == None or return_specific == 1:
        df1.TrainTestSplit()
        _, _, X_df1, wt_df1 = df1.SplitTrainTestXWts()
      if return_specific == None or return_specific == 2:
        df2.TrainTestSplit()
        _, _, X_df2, wt_df2 = df2.SplitTrainTestXWts()
    else:
      if return_specific == None or return_specific == 1: X_df1, wt_df1 = df1.SplitXWts()
      if return_specific == None or return_specific == 2: X_df2, wt_df2 = df2.SplitXWts()

    if (return_specific == None or return_specific == 1):
      if name == "B" and not all_mode:
        wt_df1 = np.multiply(wt_df1,self.model["D"].predict_reweights(X_df1.loc[:,list(self.model["D"].column_names)],cap_at=None))

      if with_reweights:
        if name == "Signal" and not all_mode:
          wt_df1 = np.multiply(np.multiply(wt_df1,self.model["C"].predict_reweights(X_df1.loc[:,list(self.model["C"].column_names)],cap_at=None)),self.model["B"].predict_reweights(X_df1))
        elif name == "B" and not all_mode:
          wt_df1 = np.multiply(wt_df1,self.model[name].predict_reweights(X_df1.loc[:,list(self.model[name].column_names)]))
        else:
          #wt_df1 = np.multiply(wt_df1,self.model[name].predict_reweights(X_df1.loc[:,list(self.model[name].column_names)],cap_at=1))
          wt_df1 = np.multiply(wt_df1,self.model[name].predict_reweights(X_df1.loc[:,list(self.model[name].column_names)],cap_at=None))
    if (return_specific == None or return_specific == 1): del df1
    if (return_specific == None or return_specific == 2): del df2 
    gc.collect()

    if not full:
      if return_specific == 1:
        return X_df1, wt_df1
      elif return_specific == "2":     
        return X_df2, wt_df2
      else:
        return X_df1, wt_df1, X_df2, wt_df2
    else:
      if return_specific == 1:
        df1 = pd.concat([X_df1,wt_df1],axis=1)
        return df1
      elif return_specific == 2:
        df2 = pd.concat([X_df2,wt_df2],axis=1)
        return df2
      else:
        return df1, df2

  def PlotReweightsFromHistograms(self, var_name, hists, divide_hists, legend_names, title_left="", title_right="", logx=[], logy=[], rebin=True, BinThreshold=100, BinUncertFraction=0.5, colours=[2,6,42,46,39,49], plot_name=""):
      c_rwt = ROOT.TCanvas('c','c',600,600)

      if rebin:
        for ind1, h1 in enumerate(hists):
          binning = FindRebinning(h1,BinThreshold=BinThreshold,BinUncertFraction=BinUncertFraction)
          for ind2, h2 in enumerate(hists):
            hists[ind2] = RebinHist(h2,binning)
        for ind_d, h_d in enumerate(divide_hists): divide_hists[ind_d] = RebinHist(divide_hists[ind_d],binning)

      for ind_d, h_d in enumerate(divide_hists):
        for i in range(0,divide_hists[ind_d].GetNbinsX()+2): divide_hists[ind_d].SetBinError(i,0)
      for ind, h in enumerate(hists): hists[ind].Divide(divide_hists[ind].Clone())

      pad_rwt = ROOT.TPad("pad_rwt","pad_rwt",0,0,1,1)
      pad_rwt.Draw()
      pad_rwt.cd()
      pad_rwt.SetBottomMargin(0.12)
      pad_rwt.SetLeftMargin(0.14)
      if var_name in logx:
        pad_rwt.SetLogx()
      if var_name in logy: pad_rwt.SetLogy()

      hists[0].SetStats(0)
      hists[0].Draw("E1")
      hists[0].SetTitle("")
      hists[0].GetXaxis().SetTitle(ReplaceName(var_name))
      hists[0].GetXaxis().SetTitleSize(0.04)
      hists[0].GetXaxis().SetTitleOffset(1.3)
      hists[0].GetYaxis().SetTitle("Average Reweight")
      hists[0].GetYaxis().SetTitleSize(0.04)
      hists[0].SetMarkerStyle(19)
      if var_name in logx:
        hists[0].GetXaxis().SetMoreLogLabels()
        hists[0].GetXaxis().SetNoExponent()

      minimum = 999999.0
      maximum = 0.0
      for ind, h in enumerate(hists):
        for i in range(0,h.GetNbinsX()+1):
          if h.GetBinError(i) != 0 and h.GetBinContent(i)-h.GetBinError(i) < minimum: minimum = h.GetBinContent(i)-h.GetBinError(i)
          if h.GetBinError(i) != 0 and h.GetBinContent(i)+h.GetBinError(i) > maximum: maximum = h.GetBinContent(i)+h.GetBinError(i)

      #hists[0].SetMinimum(0.8*minimum)
      hists[0].SetMinimum(0)
      hists[0].SetMaximum(1.4*maximum) # for legend

      for ind,h in enumerate(hists[1:]):
        hists[ind+1].Draw("E1 same")
        hists[ind+1].SetMarkerStyle(3)
        hists[ind+1].SetMarkerColor(colours[ind])
        hists[ind+1].SetLineColor(colours[ind])

      l = ROOT.TLegend(0.18,0.65,0.38,0.85);
      l.SetBorderSize(0)
      for ind,val in enumerate(hists):
        l.AddEntry(hists[ind],legend_names[ind],"lep")
      l.Draw()

      DrawTitle(pad_rwt, ReplaceName(title_left), 1, scale=0.7)
      DrawTitle(pad_rwt, ReplaceName(title_right), 3, scale=0.7)

      c_rwt.Update()
      if plot_name != "": plot_save_name = "_"+plot_name
      save_name = 'plots/reweight_ave_plot_%(var_name)s%(plot_save_name)s.pdf' % vars()
      c_rwt.SaveAs(save_name)
      print "Created", save_name
      c_rwt.Close()

      del pad_rwt, c_rwt

  def PlotClosuresFromHistograms(self, var_name, pass_hist, fail_hists, pass_hist_legend, legend_names, title_left="", title_right="", logx=[], logy=[], rebin=True, BinThreshold=100, BinUncertFraction=0.5, colours=[2,6,42,46,39,49], plot_name="",text="",extra_uncerts=None):

      if rebin:
        for ind1, h1 in enumerate(fail_hists):
          binning = FindRebinning(h1,BinThreshold=BinThreshold,BinUncertFraction=BinUncertFraction)
          for ind2, h2 in enumerate(fail_hists):
            fail_hists[ind2] = RebinHist(h2,binning)
        pass_hist = RebinHist(pass_hist,binning)
        binning = FindRebinning(pass_hist,BinThreshold=BinThreshold,BinUncertFraction=BinUncertFraction)
        pass_hist = RebinHist(pass_hist,binning)
        for ind2, h2 in enumerate(fail_hists):
          fail_hists[ind2] = RebinHist(h2,binning)
        if extra_uncerts != None:
          extra_uncerts = RebinHist(extra_uncerts,binning)


      if extra_uncerts != None:
        for j in range(0,fail_hists[0].GetNbinsX()+1):
          fail_hists[0].SetBinError(j,(fail_hists[0].GetBinError(j)**2 + (extra_uncerts.GetBinContent(j)-fail_hists[0].GetBinContent(j))**2)**0.5)


      pass_hist.Scale(1.0,"width")
      for ind,h in enumerate(fail_hists):
        fail_hists[ind].Scale(1.0,"width")


      pass_hist_divide = pass_hist.Clone()
      for i in range(0,pass_hist_divide.GetNbinsX()+2): pass_hist_divide.SetBinError(i,0)
      fail_hists_divide = []
      for ind,h in enumerate(fail_hists):
        fail_hists_divide.append(h.Clone())
        for i in range(0,fail_hists_divide[ind].GetNbinsX()+2): fail_hists_divide[ind].SetBinError(i,0)

      #pass_hist_ratio = pass_hist.Clone()
      #pass_hist_ratio.Divide(pass_hist_divide)
      pass_hist_ratio = fail_hists[0].Clone()
      pass_hist_ratio.Divide(fail_hists_divide[0])

      pass_hist_fail_hists_ratio = []
      for ind,h in enumerate(fail_hists):
        pass_hist_fail_hists_ratio.append(pass_hist.Clone())
        pass_hist_fail_hists_ratio[ind].Divide(fail_hists_divide[ind])



      c_clos = ROOT.TCanvas('c','c',600,600)

      pad1 = ROOT.TPad("pad1","pad1",0,0.37,1,1)
      pad1.SetBottomMargin(0.02)
      pad1.SetLeftMargin(0.15)
      if var_name in logx: pad1.SetLogx()
      if var_name in logy: pad1.SetLogy()
      pad1.Draw()
      pad1.cd()

      fail_hists[0].Draw("hist SAME")
      fail_hists[0].SetLineColor(1)
      fail_hists[0].SetFillColor(ROOT.kGreen+3)
      fail_hists[0].SetStats(0)
      fail_hists[0].GetXaxis().SetTitle("")
      fail_hists[0].GetYaxis().SetTitle(ReplaceName("dN/d{}".format(var_name)))
      fail_hists[0].GetYaxis().SetTitleOffset(1.2)
      fail_hists[0].GetYaxis().SetTitleSize(0.06)
      fail_hists[0].GetYaxis().SetLabelSize(0.05)
      fail_hists[0].GetXaxis().SetLabelSize(0)
      fail_hists[0].SetTitle("")

      fail_hist_uncert = fail_hists[0].Clone()
      fail_hist_uncert.SetMarkerSize(0)
      fail_hist_uncert.SetFillColorAlpha(12,0.5)
      fail_hist_uncert.SetLineWidth(0)
      fail_hist_uncert.Draw("e2 same")

      pass_hist.Draw("E SAME")
      pass_hist.SetMarkerColor(1)
      pass_hist.SetLineColor(1)
      pass_hist.SetMarkerStyle(19)

      for ind,val in enumerate(fail_hists[1:]):
        fail_hists[ind+1].Draw("SAME")
        fail_hists[ind+1].SetMarkerColor(colours[ind])
        fail_hists[ind+1].SetLineColor(colours[ind])
        fail_hists[ind+1].SetLineWidth(2)
        fail_hists[ind+1].SetMarkerStyle(2)

      minimum = 999999.0
      maximum = 0.0
      for ind, h in enumerate([pass_hist]+fail_hists):
        for i in range(0,h.GetNbinsX()+2):
          if h.GetBinError(i) != 0 and h.GetBinContent(i)-h.GetBinError(i) < minimum: minimum = h.GetBinContent(i)-h.GetBinError(i)
          if h.GetBinError(i) != 0 and h.GetBinContent(i)+h.GetBinError(i) > maximum: maximum = h.GetBinContent(i)+h.GetBinError(i)

      fail_hists[0].SetMaximum(1.6*maximum) # for legend
      fail_hists[0].GetYaxis().SetRangeUser(0,1.6*maximum)

      l = ROOT.TLegend(0.5,0.55,0.88,0.85);
      l.SetBorderSize(0)
      l.AddEntry(pass_hist,pass_hist_legend,"lep")
      l.AddEntry(fail_hists[0],legend_names[0]+" #times Fail: #chi^{2}/N_{dof} = "+str(round(pass_hist.Chi2Test(fail_hists[0],"CHI2/NDF"),2)),"f")
      for ind,h in enumerate(fail_hists[1:]):
        l.AddEntry(fail_hists[ind+1],legend_names[ind+1]+" #times Fail: #chi^{2}/N_{dof} = "+str(round(pass_hist.Chi2Test(h,"CHI2/NDF"),2)),"lep")
      l.Draw()

      #print "#chi^{2}/N_{dof} =", pass_hist.Chi2Test(fail_hists[0],"CHI2/NDF")

      latex = ROOT.TLatex()
      latex.SetNDC()
      latex.SetTextAngle(0)
      latex.SetTextColor(ROOT.kBlack)
      latex.SetTextAlign(11)
      latex.SetTextSize(0.06)
      latex.DrawLatex(0.18,0.80,text)

      c_clos.cd()
      pad2 = ROOT.TPad("pad2","pad2",0,0.05,1,0.35)
      pad2.SetLeftMargin(0.15)
      pad2.SetTopMargin(0.03)
      pad2.SetBottomMargin(0.3)
      if var_name in logx: pad2.SetLogx()
      pad2.Draw()
      pad2.cd()

      
      orig_max_ratio = max([GetMaximum(i) for i in pass_hist_fail_hists_ratio+[pass_hist_ratio]])
      orig_min_ratio = min([GetNonZeroMinimum(i) for i in pass_hist_fail_hists_ratio+[pass_hist_ratio]])
      largest_shift = max(orig_max_ratio,2-orig_min_ratio)
      possible_max = [1.2,1.4,1.6,1.8,2.0,9999.0]
      max_ratio = min(possible_max, key=lambda x:((x>largest_shift)*abs(x-largest_shift)) + ((x<=largest_shift)*9999))
      if max_ratio == 9999.0: max_ratio = 2.0
      min_ratio = max(2 - max_ratio,0)

      ratio_line = ROOT.TLine(pass_hist.GetBinLowEdge(1),1,pass_hist.GetBinLowEdge(pass_hist.GetNbinsX()+1),1)
      pass_hist_ratio.SetTitle("")
      pass_hist_ratio.SetMarkerSize(0)
      pass_hist_ratio.SetFillColorAlpha(12,0.5)
      pass_hist_ratio.SetLineWidth(0)
      pass_hist_ratio.SetAxisRange(min_ratio,max_ratio,'Y')
      pass_hist_ratio.GetYaxis().SetNdivisions(4)
      pass_hist_ratio.SetStats(0)
      pass_hist_ratio.GetXaxis().SetLabelSize(0.1)
      pass_hist_ratio.GetYaxis().SetLabelSize(0.1)
      pass_hist_ratio.GetXaxis().SetTitle(ReplaceName(var_name))
      pass_hist_ratio.GetYaxis().SetTitle("Pass / X #times Fail")
      pass_hist_ratio.GetYaxis().SetTitleColor(1)
      pass_hist_ratio.GetYaxis().SetTitleSize(0.12)
      pass_hist_ratio.GetYaxis().SetTitleOffset(0.5)
      pass_hist_ratio.GetXaxis().SetTitleSize(0.12)
      pass_hist_ratio.GetXaxis().SetTitleOffset(1.2)
      if var_name in logx:
        pass_hist_ratio.GetXaxis().SetMoreLogLabels()
        pass_hist_ratio.GetXaxis().SetNoExponent()

      pass_hist_fail_hists_ratio[0].SetMarkerColor(ROOT.kGreen+3)
      pass_hist_fail_hists_ratio[0].SetLineColor(ROOT.kGreen+3)
      pass_hist_fail_hists_ratio[0].SetMarkerStyle(19)

      for ind,h in enumerate(pass_hist_fail_hists_ratio[1:]):
        pass_hist_fail_hists_ratio[ind+1].SetLineColor(colours[ind])
        pass_hist_fail_hists_ratio[ind+1].SetMarkerColor(colours[ind])
        pass_hist_fail_hists_ratio[ind+1].SetMarkerStyle(2)

      ratio_line_up = ROOT.TLine(pass_hist.GetBinLowEdge(1),(1+max_ratio)/2,pass_hist.GetBinLowEdge(pass_hist.GetNbinsX()+1),(1+max_ratio)/2)
      ratio_line_down = ROOT.TLine(pass_hist.GetBinLowEdge(1),(1+min_ratio)/2,pass_hist.GetBinLowEdge(pass_hist.GetNbinsX()+1),(1+min_ratio)/2)
      ratio_line.SetLineStyle(3)
      ratio_line_up.SetLineStyle(3)
      ratio_line_down.SetLineStyle(3)

      pass_hist_ratio.Draw("e2")
      ratio_line.Draw("l same")
      ratio_line_up.Draw("l same")
      ratio_line_down.Draw("l same")
      for ind,h in enumerate(pass_hist_fail_hists_ratio):
        h.Draw("E same")

      DrawTitle(pad1, ReplaceName(title_left), 1, scale=1)
      DrawTitle(pad1, ReplaceName(title_right), 3, scale=1)

      if plot_name != "": plot_save_name = "_"+plot_name

      c_clos.Update()
      clos_name = 'plots/closure_plot_%(var_name)s%(plot_save_name)s.pdf' % vars()
      c_clos.SaveAs(clos_name)
      print "Created", clos_name
      c_clos.Close()

      del pad1, pad2, c_clos
      gc.collect()

  def FindBinning(self,df,col,n_bins=20,ignore_quantile=0.001,frac_to_keep=0.4):
    if len(df.loc[:,col].unique()) > n_bins: # continuous
      quant_down = df.loc[:,col].quantile(ignore_quantile)
      quant_up = df.loc[:,col].quantile(1-ignore_quantile)
      min_val = df.loc[:,col].min()
      max_val = df.loc[:,col].max()
      if ((1+frac_to_keep)*(quant_up - quant_down)) > (quant_up - min_val): quant_down = min_val 
      if ((1+frac_to_keep)*(quant_up - quant_down)) > (max_val - quant_down): 
        quant_up = max_val

      bins = list(np.linspace(quant_down,quant_up,n_bins))
    else:
      bins = list(np.linspace(df.loc[:,col].min(),df.loc[:,col].max()+1,int(df.loc[:,col].max()-df.loc[:,col].min())+2))
    return bins


  def PlotReweightsAndClosure(self, name, sel, data_type=None, title_left="", title_right="", logx=[], logy=[], compare_other_rwt=[], flat_rwt=False, plot_name="", BinThreshold=100, BinUncertFraction=0.2, add_uncerts=None):

    ROOT.gROOT.SetBatch(ROOT.kTRUE)
    ROOT.TGaxis.SetExponentOffset(-0.06, 0.01, "y");

    if type(compare_other_rwt) == str: compare_other_rwt = [compare_other_rwt]

    if name == "All":
      loop_over = []
      for i in ["B","C","D"]:
        if not (self.df1[i] == [] or self.df2[i] == []):
          loop_over.append(i)
    else:
      loop_over = [name]

    ttree_pass = {}
    ttree_fail = {}
 
    # Get binning
    df1,_,df2,_ = self.GetDataframes("All",data_type=data_type)
    df = pd.concat([df1,df2], ignore_index=True, sort=False)
    var = list(df.columns)
    binnings = [self.FindBinning(df,i) for i in var]

    for n in loop_over:

      # make dataframe fail
      df = self.GetDataframes(n,data_type=data_type,full=True,all_mode=(name=="All"),return_specific=1)

      if len(df)==0:
        print "WARNING: No events in dataset. Skipping."
        continue

      df_func = Dataframe()
      if sel != None:
        df = eval("df.loc[({})]".format(''.join(df_func.AllSplitStringsSteps(sel))))

      wt_df1 = df.loc[:,"weights"]
      X_df1 = df.drop(["weights"],axis=1)

      var_df = X_df1.loc[:,list(set(var+compare_other_rwt))]

      df1 = pd.concat([wt_df1,var_df],axis=1)

      if name == "B":
        df1.loc[:,"reweights"] = self.model[name].predict_reweights(X_df1.loc[:,list(self.model[name].column_names)])
      else:
        df1.loc[:,"reweights"] = self.model[name].predict_reweights(X_df1.loc[:,list(self.model[name].column_names)],cap_at=None)

      if add_uncerts != None:
        tmp_df = GetNonClosureLargestShift(X_df1.loc[:,list(self.model[name].column_names)],add_uncerts,df1.loc[:,"reweights"])
        df1.loc[:,"uncerts_up"] = tmp_df.loc[:,"up"]
        df1.loc[:,"uncerts_down"] = tmp_df.loc[:,"down"]
        del tmp_df

      # load to root ttree pass
      df_write = Dataframe()
      df_write.dataframe = df1
      ttree_fail[n] = df_write.WriteToRoot(None,key='ntuple',return_tree=True)

      del df_write, df1, wt_df1, X_df1, var_df, df
      gc.collect()
 
      # make dataframe pass
      df = self.GetDataframes(n,data_type=data_type,full=True,all_mode=(name=="All"),return_specific=2)

      if len(df)==0:
        print "WARNING: No events in dataset. Skipping."
        continue

      df_func = Dataframe()
      if sel != None:
        df = eval("df.loc[({})]".format(''.join(df_func.AllSplitStringsSteps(sel))))

      wt_df1 = df.loc[:,"weights"]
      X_df1 = df.drop(["weights"],axis=1)
      var_df = X_df1.loc[:,list(set(var+compare_other_rwt))]

      df1 = pd.concat([wt_df1,var_df],axis=1)

      #if name == "B":
      #  df1.loc[:,"reweights"] = self.model[name].predict_reweights(X_df1.loc[:,list(self.model[name].column_names)])
      #else:
      #  df1.loc[:,"reweights"] = self.model[name].predict_reweights(X_df1.loc[:,list(self.model[name].column_names)],cap_at=None)

      # load to root ttree fail
      df_write = Dataframe()
      df_write.dataframe = df1
      ttree_pass[n] = df_write.WriteToRoot(None,key='ntuple',return_tree=True)

      del df_write, df1, wt_df1, X_df1, var_df, df
      gc.collect()

    var = [i.replace("(","_").replace("*","_times_").replace("/","_divide_").replace(")","") for i in var]
    # loop through variables
    for var_ind, var_name in enumerate(var):
      ### Reweight Plots ###

      var_binning = binnings[var_ind]
      if isinstance(var_binning,tuple):
        bins = array('f', map(float,[(float(var_binning[2]-var_binning[1])/var_binning[0])*i + float(var_binning[1]) for i in range(0,var_binning[0])]))
      elif isinstance(var_binning,list):
        bins = array('f', map(float,var_binning))

      hout = ROOT.TH1D('hout','',len(bins)-1, bins)
      for i in range(0,hout.GetNbinsX()+2): hout.SetBinError(i,0)
      hout.SetDirectory(ROOT.gROOT)

      hist1 = []
      hist2 = []
      for indh, nh in enumerate(loop_over):
        hist1.append(hout.Clone())
        hist1[indh].SetName('hist1_%(indh)i' % vars())

        hist2.append(hout.Clone())
        hist2[indh].SetName('hist2_%(indh)i' % vars())

        hist1[indh].SetDirectory(ROOT.gROOT)
        ttree_fail[nh].Draw('%(var_name)s>>hist1_%(indh)i' % vars(),'weights*(1)','goff')
        hist1[indh] = ttree_fail[nh].GetHistogram()
 
        hist2[indh].SetDirectory(ROOT.gROOT)
        ttree_fail[nh].Draw('%(var_name)s>>hist2_%(indh)i' % vars(),'weights*reweights*(1)','goff')
        hist2[indh] = ttree_fail[nh].GetHistogram()

      hist3 = []
      for ind,val in enumerate(compare_other_rwt):
        hist3.append(hout.Clone())
        hist3[ind].SetName('hist3_%(val)s' % vars())
 
        hist3[ind].SetDirectory(ROOT.gROOT)
        ttree_fail["C"].Draw('%(var_name)s>>hist3_%(val)s' % vars(),'weights*%(val)s*(1)' % vars(),'goff')
        hist3[ind] = ttree_fail["C"].GetHistogram()
        hist1.append(copy.deepcopy(hist1[0]))

      non_zero_bins = 0
      for i in range(0,hist1[0].GetNbinsX()+1):
        if hist1[0].GetBinContent(i)>0.0: non_zero_bins += 1
      rebin = (non_zero_bins > 10) 
      if "deepTau" in var_name: rebin = False

      self.PlotReweightsFromHistograms(var_name, hist2+hist3, hist1, loop_over+compare_other_rwt, title_left=title_left, title_right=title_right, logx=logx, logy=logy, rebin=rebin, BinThreshold=BinThreshold, BinUncertFraction=BinUncertFraction, plot_name=plot_name)


      for ind, n in enumerate(loop_over):
        hist1 = hout.Clone()
        hist1.SetName('hist1')
        hist2 = hout.Clone()
        hist2.SetName('hist2')

        hist3 = []
        if n == "C":
          for ind,val in enumerate(compare_other_rwt):
            hist3.append(hout.Clone())
            hist3[ind].SetName('hist3_%(val)s' % vars())
        if flat_rwt:
          hist4 = hout.Clone()
          hist4.SetName('hist4')

        if n not in ttree_pass or n not in ttree_fail:
          print "WARNING: No events in dataset. Skipping."
          continue

        hist1.SetDirectory(ROOT.gROOT) 
        ttree_pass[n].Draw('%(var_name)s>>hist1' % vars(),'weights*(1)','goff')
        hist1 = ttree_pass[n].GetHistogram()

        hist2.SetDirectory(ROOT.gROOT)
        ttree_fail[n].Draw('%(var_name)s>>hist2' % vars(),'weights*reweights*(1)','goff')
        hist2 = ttree_fail[n].GetHistogram()

        if add_uncerts != None:

          hist2_up = hout.Clone()
          hist2_up.SetName('hist2_up')

          hist2_down = hout.Clone()
          hist2_down.SetName('hist2_down')

          hist2_up.SetDirectory(ROOT.gROOT)
          ttree_fail[n].Draw('%(var_name)s>>hist2_up' % vars(),'weights*uncerts_up*(1)','goff')
          hist2_up = ttree_fail[n].GetHistogram()
  
          hist2_down.SetDirectory(ROOT.gROOT)
          ttree_fail[n].Draw('%(var_name)s>>hist2_down' % vars(),'weights*uncerts_down*(1)','goff')
          hist2_down = ttree_fail[n].GetHistogram()
        else:
          hist2_up = None

        hist4_add = []
        leg_list = []
        if n == "C":
          for ind,val in enumerate(compare_other_rwt):
            hist3[ind].SetDirectory(ROOT.gROOT)
            ttree_fail[n].Draw('%(var_name)s>>hist3_%(val)s' % vars(),'weights*%(val)s*(1)' % vars(),'goff')
            hist3[ind] = ttree_fail[n].GetHistogram()
        if flat_rwt:
          norm_wt = self.model[name].initial_normalization
          hist4.SetDirectory(ROOT.gROOT)
          ttree_fail[n].Draw('%(var_name)s>>hist4' % vars(),'weights*%(norm_wt)s*(1)' % vars(),'goff')
          hist4 = ttree_fail[n].GetHistogram()
          if n == "C": leg_list = compare_other_rwt + ["Norm"]
          else: leg_list = ["Norm"]
          hist4_add = [hist4]
        if name == "All":
          closure_name = plot_name+"_"+n.lower().replace(" ","_").replace("_{","").replace("}","")
        else:
          closure_name = plot_name
        self.PlotClosuresFromHistograms(var_name, hist1, [hist2]+hist3+hist4_add, "Pass", ["BDT F_{F}"]+leg_list, title_left=title_left, title_right=title_right, logx=logx, logy=logy, rebin=rebin, BinThreshold=BinThreshold, BinUncertFraction=BinUncertFraction, plot_name=closure_name, text="Region "+n, extra_uncerts=hist2_up)

      if name == "All":
        # Do combined closure plot
        hist1_total = hout.Clone()
        hist1_total.SetName('hist1_total')
        hist2_total = hout.Clone()
        hist2_total.SetName('hist2_total')
        if add_uncerts != None:
          hist2_total_up = hout.Clone()
          hist2_total_up.SetName('hist2_total_up')
        else:
          hist2_total_up = None
        for n in loop_over:
          hist1_temp = hout.Clone()
          hist1_temp.SetName('hist1_temp')
          hist2_temp = hout.Clone()
          hist2_temp.SetName('hist2_temp')
          if add_uncerts != None:
            hist2_temp_up = hout.Clone()
            hist2_temp_up.SetName('hist2_temp_up')
          if n in ttree_pass:
            hist1_temp.SetDirectory(ROOT.gROOT)
            ttree_pass[n].Draw('%(var_name)s>>hist1_temp' % vars(),'weights*(1)','goff')
            hist1_temp = ttree_pass[n].GetHistogram()
            hist1_total.Add(hist1_temp)
          if n in ttree_fail:
            hist2_temp.SetDirectory(ROOT.gROOT)
            ttree_fail[n].Draw('%(var_name)s>>hist2_temp' % vars(),'weights*reweights*(1)','goff')
            hist2_temp = ttree_fail[n].GetHistogram()
            hist2_total.Add(hist2_temp)
            if add_uncerts != None:
              hist2_temp_up.SetDirectory(ROOT.gROOT)
              ttree_fail[n].Draw('%(var_name)s>>hist2_temp_up' % vars(),'weights*uncerts_up*(1)','goff')
              hist2_temp_up = ttree_fail[n].GetHistogram()
              hist2_total_up.Add(hist2_temp_up)


        closure_name = plot_name+"_all"
        self.PlotClosuresFromHistograms(var_name, hist1_total, [hist2_total], "Pass", ["BDT F_{F}"], title_left=title_left, title_right=title_right, logx=logx, logy=logy, rebin=rebin, BinThreshold=BinThreshold, BinUncertFraction=BinUncertFraction, plot_name=closure_name, text="Regions B, C and D", extra_uncerts=hist2_total_up)

      

