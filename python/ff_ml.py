from UserCode.BDTFakeFactors import reweighter
from UserCode.BDTFakeFactors.Dataframe import Dataframe
from UserCode.BDTFakeFactors.plotting import DrawClosurePlots, DrawColzReweightPlots
import copy
import pandas as pd
import numpy as np
from UserCode.BDTFakeFactors import reweighter

class ff_ml():

  def __init__(self):
    self.df1 = {"Raw F_{F}":[], "Alternative F_{F}":[], "Correction":[], "Signal":[]}
    self.df2 = {"Raw F_{F}":[], "Alternative F_{F}":[], "Correction":[], "Signal":[]}
    self.model = {"Raw F_{F}":None, "Alternative F_{F}":None, "Correction":None, "Signal":None}

  def AddDataframes(self,df1, df2, name):
    if name not in ["Raw F_{F}","Alternative F_{F}","Correction","Signal"]:
      print "ERROR: Invalid name given. Please give Raw, Alternative, Correction or Signal"
    else:
      if df1 != None:
        self.df1[name].append(copy.deepcopy(df1))
      if df2 != None:
        self.df2[name].append(copy.deepcopy(df2))

  def AddModel(self,model,name):
    if name not in ["Raw F_{F}","Alternative F_{F}","Correction","Signal"]:
      print "ERROR: Invalid name given. Please give Raw, Alternative, Correction or Signal"
    else:
      self.model[name] = copy.deepcopy(model)


  def GetDataframes(self,name,data_type=None,with_reweights=False,full=False):
    if name not in ["Raw F_{F}","Alternative F_{F}","Correction","Signal"]:
      print "ERROR: Invalid name given. Please give Raw, Alternative, Correction or Signal"
    else:
      df1 = Dataframe()
      for ind1,i1 in enumerate(self.df1[name]):
        if ind1 == 0:
          df1.dataframe = pd.read_pickle(i1)
        else:
          df1.dataframe  = pd.concat([df1.dataframe,pd.read_pickle(i1)], ignore_index=True, sort=False) 

      df2 = Dataframe()
      for ind2,i2 in enumerate(self.df2[name]):
        if ind2 == 0:
          df2.dataframe = pd.read_pickle(i2)
        else:
          df2.dataframe  = pd.concat([df2.dataframe,pd.read_pickle(i2)], ignore_index=True, sort=False)
 
    if data_type == "train":
      df1.TrainTestSplit()
      df2.TrainTestSplit()
      X_df1, wt_df1, _, _ = df1.SplitTrainTestXWts()
      X_df2, wt_df2, _, _ = df2.SplitTrainTestXWts()
    elif data_type == "test":
      df1.TrainTestSplit()
      df2.TrainTestSplit()
      _, _, X_df1, wt_df1 = df1.SplitTrainTestXWts()
      _, _, X_df2, wt_df2 = df2.SplitTrainTestXWts()
    else:
      X_df1, wt_df1 = df1.SplitXWts()
      X_df2, wt_df2 = df2.SplitXWts()

    if name == "Correction":
      wt_df1 = np.multiply(wt_df1,self.model["Alternative F_{F}"].predict_reweights(X_df1))

    if with_reweights:
      if name == "Signal":
        wt_df1 = np.multiply(np.multiply(wt_df1,self.model["Raw F_{F}"].predict_reweights(X_df1)),self.model["Correction"].predict_reweights(X_df1))
      else:
        wt_df1 = np.multiply(wt_df1,self.model[name].predict_reweights(X_df1))

    del df1
    del df2 

    if not full:
      return X_df1, wt_df1, X_df2, wt_df2
    else:
      df1 = pd.concat([X_df1,wt_df1],axis=1) 
      df2 = pd.concat([X_df2,wt_df2],axis=1) 
      return df1, df2


  def PlotReweights(self, var, name, sel, data_type=None, title_left="", title_right="", plot_name="reweight_colz_plot"):

    df, _ = self.GetDataframes(name,data_type=data_type,full=True)
    df_func = Dataframe()
    if sel != None:
      df = eval("df.loc[({})]".format(''.join(df_func.AllSplitStringsSteps(sel))))

    X_df1, wt_df1, _, _ = self.GetDataframes(name,data_type=data_type)
    reweights = pd.Series(self.model[name].predict_reweights(X_df1),name="reweights")

    var_df = X_df1.loc[:,var]
    DrawColzReweightPlots(var_df, reweights, wt_df1, var, name, plot_name=plot_name, title_left=title_left, title_right=title_right)

  def PlotClosure(self, var, name, sel, data_type=None, title_left="", title_right="", plot_name="closure_colz_plot", fillcolour=38):
    df_func = Dataframe()
    df,_ = self.GetDataframes(name,data_type=data_type,full=True,with_reweights=(self.model[name]!=None))
    if sel != None: df = eval("df.loc[({})]".format(''.join(df_func.AllSplitStringsSteps(sel))))
    df1 = copy.deepcopy(df)
    _,df = self.GetDataframes(name,data_type=data_type,full=True,with_reweights=(self.model[name]!=None))
    if sel != None: df = eval("df.loc[({})]".format(''.join(df_func.AllSplitStringsSteps(sel))))
    df2 = copy.deepcopy(df)

    del df
    cut_df1 = df1.loc[:,["weights",var[0]]]
    cut_df2 = df2.loc[:,["weights",var[0]]]
    del df1
    del df2
    DrawClosurePlots(cut_df1, cut_df2, "Reweight x fail", "pass", var[0], var[1], plot_name=plot_name, title_left=title_left, title_right=title_right, fillcolour=fillcolour)

