from collections import OrderedDict
import pandas as pd
import numpy as np
import copy
import json

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

  print pass_val
  print fail_val

  sel_string = "("
  for ind, v in enumerate(var):
    if not continuous:
      sel_string += "("
      for pv in pass_val[ind]:
        sel_string += "(df.{} == {})".format(v,pv)
        if pv != pass_val[ind][-1]:
          sel_string += ") | "
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
  max_uncert = max_uncert * reweights
  return_df = pd.DataFrame()
  return_df.loc[:,"up"] = reweights + max_uncert
  return_df.loc[:,"down"] = reweights - max_uncert
  print "Non Closure Average Shift:", (return_df.loc[:,"up"].sum()/reweights.sum())
  return return_df
