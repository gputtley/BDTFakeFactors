import ROOT
import os
from UserCode.BDTFakeFactors.batch import CreateBatchJob,SubmitBatchJob
import argparse

#python scripts/batch_add_to_trees.py --input_location=/vols/cms/gu18/Offline/output/MSSM/vlq_2018_bkg_data/ --output_location=/vols/cms/gu18/Offline/output/MSSM/vlq_2018_bdt
#python scripts/batch_add_to_trees.py -input_location=/vols/cms/gu18/Offline/output/MSSM/vlq_2018_pre_bdt --output_location=/vols/cms/gu18/Offline/output/MSSM/vlq_2018_bdt_v2 --syst_trees --hadd
#python scripts/batch_add_to_trees.py --input_location=/vols/cms/gu18/Offline/output/4tau/2018_1907 --output_location=/vols/cms/gu18/Offline/output/4tau/2018_1907_ff

parser = argparse.ArgumentParser()
parser.add_argument('--analysis',help= 'Analysis to train BDT for', default='4tau')
parser.add_argument('--pass_wp',help= 'Pass WP for fake factors', default='loose')
parser.add_argument('--fail_wp',help= 'Channel to train BDT for', default=None)
parser.add_argument('--input_location',help= 'Name of input location (not including file name)', default='/vols/cms/gu18/Offline/output/MSSM/vlq_2018_matched_v2/')
parser.add_argument('--output_location',help= 'Name of output location (not including file name)', default='/vols/cms/gu18/Offline/output/MSSM/vlq_2018_bdt/')
parser.add_argument('--file',help= 'Specify to run for a specific file', default='all')
parser.add_argument('--channel',help= 'Limit to single channel. If not set will do all', default=None)
parser.add_argument("--syst_trees", action='store_true',help="Also run on systematic variation trees")
parser.add_argument("--hadd", action='store_true',help="Hadd files after adding branch")
parser.add_argument("--only_data", action='store_true',help="Only run for data files")
args = parser.parse_args()

loc = args.input_location
newloc = args.output_location
analysis = args.analysis
pass_wp=args.pass_wp
fail_wp=args.fail_wp

if loc[-1] == "/": loc = loc[:-1]
if newloc[-1] == "/": newloc = newloc[:-1]

data_files = ["SingleMuon","SingleElectron","Tau","EGamma"]

splitting = 100000 

cmssw_base = os.getcwd().replace('src/UserCode/sig_vs_bkg_discriminator','')

#if not args.hadd and args.file=="all" and len(os.listdir('jobs')) != 0: os.system("rm jobs/*")

if not os.path.isdir(newloc): os.system("mkdir "+newloc)


for file_name in os.listdir(loc):
  if "ZstarTophi" in file_name: continue
  if ".root" in file_name  and (args.file=="all" or args.file==file_name):
    if args.only_data:
      b = True
      for dfs in data_files:
        if file_name.startswith(dfs): b = False
      if b: continue      

    if "_et_" in file_name:
      channel = "et"
    elif "_mt_" in file_name:
      channel = "mt"
    elif "_tt_" in file_name:
      channel = "tt"
    elif "_em_" in file_name:
      channel = "em"
    elif "_zmm_" in file_name:
      channel = "zmm"
    elif "_tttt_" in file_name:
      channel = "tttt"
    elif "_ettt_" in file_name:
      channel = "ettt"
    elif "_mttt_" in file_name:
      channel = "mttt"
    elif "_emtt_" in file_name:
      channel = "emtt"
    elif "_mmtt_" in file_name:
      channel = "mmtt"
    elif "_eett_" in file_name:
      channel = "eett"
    elif "_ttt_" in file_name:
      channel = "ttt"
    else:
      continue

    if "_2016_preVFP" in file_name:
      year = "2016_preVFP"
    elif "_2016_postVFP" in file_name:
      year = "2016_postVFP"
    elif "_2016" in file_name:
      year = "2016"
    elif "_2017" in file_name:
      year = "2017"
    elif "_2018" in file_name:
      year = "2018"


    if not "_"+channel+"_"+year+".root" in file_name: continue

    if not args.hadd and (args.channel == None or args.channel == channel):
      f = ROOT.TFile(loc+'/'+file_name,"READ")
      t = f.Get("ntuple")
      if type(t) == type(ROOT.TObject()): continue
      ent = t.GetEntries()
      splits = ((ent - (ent%splitting))/splitting) + 1

      for i in range(0,splits):
        cmd = "python scripts/add_ff_to_trees.py --input_location=%(loc)s --output_location=%(newloc)s --filename=%(file_name)s --channel=%(channel)s --year=%(year)s --splitting=%(splitting)i --offset=%(i)i  --analysis=%(analysis)s --fail_wp=%(fail_wp)s --pass_wp=%(pass_wp)s" % vars()
        CreateBatchJob('jobs/'+file_name.replace('.root','_'+str(i)+'.sh'),cmssw_base,[cmd])
        SubmitBatchJob('jobs/'+file_name.replace('.root','_'+str(i)+'.sh'),time=180,memory=24,cores=1)

    else:
      if (args.channel == None or args.channel == channel):
        cmd1 = "hadd -f {}/{} {}/{}".format(newloc,file_name,newloc,file_name.replace(".root","_*"))
        cmd2 = "rm {}/{}".format(newloc,file_name.replace(".root","_*"))
        cmd3 = "echo 'Finished processing'"
        CreateBatchJob('jobs/hadd_'+file_name.replace('.root','.sh'),cmssw_base,[cmd1,cmd2,cmd3])
        SubmitBatchJob('jobs/hadd_'+file_name.replace('.root','.sh'),time=180,memory=24,cores=1)

  elif args.syst_trees:

    for syst_file_name in os.listdir(loc+"/"+file_name):

      if not os.path.isdir(newloc+"/"+file_name): os.system("mkdir "+newloc+"/"+file_name)

      if ".root" in syst_file_name  and (args.file=="all" or args.file==syst_file_name):

        if args.only_data:
          b = True
          for dfs in data_files:
            if file_name.startswith(dfs): b = False
          if b: continue

        if "_et_" in syst_file_name:
          channel = "et"
        elif "_mt_" in syst_file_name:
          channel = "mt"
        elif "_tt_" in syst_file_name:
          channel = "tt"
        elif "_em_" in syst_file_name:
          channel = "em"
        elif "_tttt_" in file_name:
          channel = "tttt"
        elif "_ettt_" in file_name:
          channel = "ettt"
        elif "_mttt_" in file_name:
          channel = "mttt"
        elif "_emtt_" in file_name:
          channel = "emtt"
        elif "_mmtt_" in file_name:
          channel = "mmtt"
        elif "_eett_" in file_name:
          channel = "eett"
        elif "_ttt_" in file_name:
          channel = "ttt"
        else:
          continue


        if "_2016_preVFP" in syst_file_name:
          year = "2016_preVFP"
        elif "_2016_postVFP" in syst_file_name:
          year = "2016_postVFP"
        elif "_2017" in syst_file_name:
          year = "2017"
        elif "_2018" in syst_file_name:
          year = "2018"
    
    
        if not args.hadd and (args.channel == None or args.channel == channel):
          f = ROOT.TFile(loc+'/'+file_name+'/'+syst_file_name,"READ")
          t = f.Get("ntuple")
          if str(type(t)) == "<class 'ROOT.TObject'>":continue
          ent = t.GetEntries()
          splits = ((ent - (ent%splitting))/splitting) + 1
    
          for i in range(0,splits):
            cmd = "python scripts/add_ff_to_trees.py --input_location=%(loc)s/%(file_name)s --output_location=%(newloc)s/%(file_name)s --filename=%(syst_file_name)s --channel=%(channel)s --year=%(year)s --splitting=%(splitting)i --offset=%(i)i --analysis=%(analysis)s --fail_wp=%(fail_wp)s --pass_wp=%(pass_wp)s" % vars()
            CreateBatchJob('jobs/'+syst_file_name.replace('.root',"_")+file_name+'_'+str(i)+'.sh',cmssw_base,[cmd])
            SubmitBatchJob('jobs/'+syst_file_name.replace('.root',"_")+file_name+'_'+str(i)+'.sh',time=180,memory=24,cores=1)
    
        else:
          if (args.channel == None or args.channel == channel):
            cmd1 = "hadd -f {}/{}/{} {}/{}/{}".format(newloc,file_name,syst_file_name,newloc,file_name,syst_file_name.replace(".root","_*"))
            cmd2 = "rm {}/{}/{}".format(newloc,file_name,syst_file_name.replace(".root","_*"))
            cmd3 = "echo 'Finished processing'"
            CreateBatchJob('jobs/hadd_'+syst_file_name.replace('.root',"_")+file_name+'.sh',cmssw_base,[cmd1,cmd2,cmd3])
            SubmitBatchJob('jobs/hadd_'+syst_file_name.replace('.root',"_")+file_name+'.sh',time=180,memory=24,cores=1)

