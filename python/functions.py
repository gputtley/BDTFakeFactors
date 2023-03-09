from collections import OrderedDict
import pandas as pd
import numpy as np
import copy
import json

def GetNonZeroMinimum(hist):
  no_min = True
  for i in range(0,hist.GetNbinsX()+1):
    bc = hist.GetBinContent(i)
    if no_min and bc != 0: 
      min_bin = bc*1.0
      no_min = False
    elif not no_min and bc != 0:
      if bc < min_bin:
        min_bin = bc*1.0
  if no_min: min_bin = 0.0
  return min_bin



def EqualBinStats(dataframe,weights,col,n_bins=20,min_bin=None,max_bin=None,ignore_quantile=0.0):
  df = copy.deepcopy(dataframe)
  df.loc[:,"weight"] = weights
  if len(df.loc[:,col].unique()) > n_bins: # continuous
    quant_down = df.loc[:,col].quantile(ignore_quantile)
    quant_up = df.loc[:,col].quantile(1-ignore_quantile)
    if max_bin == None:
      end = quant_up
    else:
      end = min(max_bin,quant_up)

    if min_bin == None:
      start = quant_down
    else:
      start = max(min_bin,quant_down)

    bins = list(np.linspace(start,end,n_bins))
  else:
    bins = sorted(df.loc[:,col].unique())
    bins.append(2*bins[-1]-bins[-2])

  bin_weight_sum =  [df.loc[((df.loc[:,col]>=bins[i])&(df.loc[:,col]>=bins[i+1])),"weight"].sum() for i in range(0,len(bins)-1)]
  scale_by = [max(bin_weight_sum)/i for i in bin_weight_sum]
  bins[-1] = "999999"
  bins[0] = "-999999"
  join_str = ["(((x>={}) * (x<{})) * ({}))".format(bins[i],bins[i+1],scale_by[i]) for i in range(0,len(bins)-1)]
  return " + ".join(join_str)

def PrintDatasetSummary(name,dataset):
  print name
  print dataset.head(10)
  if "weights" in dataset.columns:
    print "Total Length = {}, Total Sum of Weights = {}".format(len(dataset),dataset.loc[:,"weights"].sum())
    print "Average Weights = {}".format(dataset.loc[:,"weights"].sum()/len(dataset))
    print ""
  else:
    print "Total Length = {}".format(len(dataset))
    print ""

def AddColumnsToDataframe(df, columns, vals):
  for ind, i in enumerate(columns):
    df.loc[:,i] = val[ind]
  return df.reindex(sorted(df.columns), axis=1)

def MakeSelFromList(lst,use="and"):
  if "" in lst: lst.remove("")
  lst.sort()
  sel = ""
  for ind, l in enumerate(lst):
    sel += l
    if ind != len(lst)-1:
      if use == "and":
        sel += " && "
      elif use == "or":
        sel += " || "
  return sel

def MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL):
  f_i = []
  p_i = []
  for ind_t, t in enumerate(t_ff):
    f_i.append(FAIL.replace("X",str(t)))
    p_i.append(PASS.replace("X",str(t))) 

  o_ff = tuple(set(lst_n) - set(t_ff))
  f_o = []
  p_o = []
  for ind_o, o in enumerate(o_ff):
    f_o.append(FAIL.replace("X",str(o)))
    p_o.append(PASS.replace("X",str(o)))

  sel = OrderedDict()

  sel["Raw F_{F}"] = {
                      "pass":MakeSelFromList([SIDEBAND]+p_i+p_o),
                      "fail":MakeSelFromList([SIDEBAND]+f_i+p_o)
                      }

  sel["Alternative F_{F}"] = {
                              "pass":"((" + MakeSelFromList([SIDEBAND]+p_i) + ") && (" + MakeSelFromList(f_o,use="or") + "))",
                              "fail":"((" + MakeSelFromList([SIDEBAND]+f_i) + ") && (" + MakeSelFromList(f_o,use="or") + "))"
                              }
  sel["Correction"] = {
                       "pass":"((" + MakeSelFromList([SIGNAL]+p_i) + ") && (" + MakeSelFromList(f_o,use="or") + "))",
                       "fail":"((" + MakeSelFromList([SIGNAL]+f_i) + ") && (" + MakeSelFromList(f_o,use="or") + "))"
                       }

  sel["Signal"] = {
                       "pass":MakeSelFromList([SIGNAL]+p_i+p_o),
                       "fail":MakeSelFromList([SIGNAL]+f_i+p_o)
                       }

  return sel

def SampleValuesAndGiveLargestShift(df,model,var,pass_val,fail_val,continuous=False,n_samples=5,name=""):

  if type(var) == str: var = [var]
  if type(pass_val[0]) != list: pass_val = [pass_val]
  if type(fail_val[0]) != list: fail_val = [fail_val]

  sel_string = "("
  for ind, v in enumerate(var):
    if not continuous:
      sel_string += "("
      for pv in pass_val[ind]:
        sel_string += "(df.{} == {})".format(v,pv)
        if pv != pass_val[ind][-1]:
          sel_string += " | "
        else:
          sel_string += ")"
    else:
      sel_string += "((df.{} >= {}) & (df.{} < {}))".format(v,pass_val[ind][0],v,pass_val[ind][1])

    if v != var[-1]: 
      sel_string += "& "
    else:
      sel_string += ")"

  store_df = copy.deepcopy(df)
  diff = pd.DataFrame()
  original_weights = model.predict_reweights(df)
  for i in range(1,n_samples+1):
    df = copy.deepcopy(store_df)
    for ind, v in enumerate(var):   
      if not continuous:
        exec('df.loc[{},v] = np.random.choice(fail_val[ind], df.loc[{},:].shape[0])'.format(sel_string,sel_string))
      else:
        exec('df.loc[{},v] = np.random.uniform(fail_val[ind][0],fail_val[ind][1],df.loc[{},:].shape[0])'.format(sel_string,sel_string))
    diff.loc[:,i] = abs(original_weights - model.predict_reweights(df))
  max_diff = diff.max(numeric_only = True, axis = 1)
  return_df = pd.DataFrame()
  return_df.loc[:,"up"] = max_diff + original_weights
  return_df.loc[:,"down"] = original_weights - max_diff
  return_df.loc[(return_df.loc[:,"down"]<0),"down"] = 0.0
  return_df.loc[(return_df.loc[:,"up"]<0),"up"] = 0.0
  print var, "Average Shift:", (return_df.loc[:,"up"].sum()/original_weights.sum())
  return return_df


def GetNonClosureLargestShift(df,json_file,reweights):
  with open(json_file) as json_file: l_funcs = json.load(json_file,object_pairs_hook=OrderedDict)
  uncert_df = pd.DataFrame()
  i = 0
  for k,v in l_funcs.iteritems():
    exec("uncert_df.loc[:,k] = df.iloc[:,i].apply(lambda x: {})").format(v)
    i += 1
  max_uncert = uncert_df.max(numeric_only = True, axis = 1)
  #print pd.concat([uncert_df,max_uncert],axis=1)
  max_uncert = max_uncert * reweights
  return_df = pd.DataFrame()
  return_df.loc[:,"up"] = reweights + max_uncert
  return_df.loc[:,"down"] = reweights - max_uncert
  return_df.loc[(return_df.loc[:,"down"]<0),"down"] = 0.0
  return_df.loc[(return_df.loc[:,"up"]<0),"up"] = 0.0
  
  print "Non Closure Average Shift:", (return_df.loc[:,"up"].sum()/reweights.sum())
  return return_df
