COUNT=0
TOTAL=0

for i in $(ls jobs/*output.log); do 
  (( TOTAL++ ))
  if ! grep -q "Finished processing" $i ; then
    echo Error found for $i
    job=${i//_output.log/".sh"}
    new_err=${i//_output.log/"_error.log"}
    new_out=${i//_output.log/"_output.log"}
    #echo $job 
    if [[ $1 != "" ]]; then
      qsub -e $new_err -o $new_out -V -q hep.q -l h_rt=3:0:0 -l h_vmem=24G -cwd $job
    fi
    (( COUNT++ ))
  fi

done
echo $COUNT jobs out of $TOTAL failed
