from collections import OrderedDict
import pandas as pd
import numpy as np

def PrintDatasetSummary(name,dataset):
  print name
  print dataset.head(10)
  print "Total Length = {}, Total Sum of Weights = {}".format(len(dataset),dataset.loc[:,"weights"].sum())
  print "Average Weights = {}".format(dataset.loc[:,"weights"].sum()/len(dataset))
  print ""

def AddColumnsToDataframe(df, columns, vals):
  for ind, i in enumerate(columns):
    df.loc[:,i] = val[ind]
  return df.reindex(sorted(df.columns), axis=1)

def MakeSelFromList(lst):
  if "" in lst: lst.remove("")
  lst.sort()
  sel = ""
  for ind, l in enumerate(lst):
    sel += l
    if ind != len(lst)-1:
      sel += " && "
  return sel

def MakeSelDict(t_ff,lst_n,SIDEBAND,SIGNAL,PASS,FAIL):
  f_i = ""
  p_i = ""
  for ind_t, t in enumerate(t_ff):
    f_i += FAIL.replace("X",str(t))
    p_i += PASS.replace("X",str(t))
    if ind_t != len(t_ff)-1:
      f_i += " && "
      p_i += " && "

  o_ff = tuple(set(lst_n) - set(t_ff))
  f_o = ""
  p_o = ""
  for ind_o, o in enumerate(o_ff):
    f_o += FAIL.replace("X",str(o))
    p_o += PASS.replace("X",str(o))
    if ind_o != len(o_ff)-1:
      f_o += " && "
      p_o += " && "

  sel = OrderedDict()

  sel["Raw F_{F}"] = {
                      "pass":MakeSelFromList([SIDEBAND,p_i,p_o]),
                      "fail":MakeSelFromList([SIDEBAND,f_i,p_o])
                      }

  sel["Alternative F_{F}"] = {
                              "pass":MakeSelFromList([SIDEBAND,p_i,f_o]),
                              "fail":MakeSelFromList([SIDEBAND,f_i,f_o])
                              }
  sel["Correction"] = {
                       "pass":MakeSelFromList([SIGNAL,p_i,f_o]),
                       "fail":MakeSelFromList([SIGNAL,f_i,f_o])
                       }

  sel["Signal"] = {
                       "pass":MakeSelFromList([SIGNAL,p_i,p_o]),
                       "fail":MakeSelFromList([SIGNAL,f_i,p_o])
                       }
  return sel

