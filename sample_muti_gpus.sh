
mode="sty_concat_sigma_cosine_vpem"   
OPENAI_LOGDIR="./checkpoint/vein_${mode}"        
model_path="./checkpoint/vein_${mode}/ema_0.9999_070000.pt"
sams=20             # Number of samples for each ID.
batch_size=100   
num_samples=10000   # Number of images / Number of gpus
name=palmvein
conddir=./pvtree/pv_pattern_results

# To speed up generation, the dataset is split into multiple subsets based on the 
# number of GPUs, and generation tasks are executed in parallel on separate GPUs.
# '-n' means 8 gpus

python3 splitdata.py \
   --p $conddir/$name \
   --n 8

wait

cond_dir0=${conddir}/${name}_0
cond_dir1=${conddir}/${name}_1
cond_dir2=${conddir}/${name}_2
cond_dir3=${conddir}/${name}_3
cond_dir4=${conddir}/${name}_4
cond_dir5=${conddir}/${name}_5
cond_dir6=${conddir}/${name}_6
cond_dir7=${conddir}/${name}_7

outname=./results_${name}_${mode}

bash sample.sh $OPENAI_LOGDIR $batch_size $num_samples $model_path $outname $cond_dir0 $sams 0 &
bash sample.sh $OPENAI_LOGDIR $batch_size $num_samples $model_path $outname $cond_dir1 $sams 1 &
bash sample.sh $OPENAI_LOGDIR $batch_size $num_samples $model_path $outname $cond_dir2 $sams 2 &
bash sample.sh $OPENAI_LOGDIR $batch_size $num_samples $model_path $outname $cond_dir3 $sams 3 &
bash sample.sh $OPENAI_LOGDIR $batch_size $num_samples $model_path $outname $cond_dir4 $sams 4 &
bash sample.sh $OPENAI_LOGDIR $batch_size $num_samples $model_path $outname $cond_dir5 $sams 5 &
bash sample.sh $OPENAI_LOGDIR $batch_size $num_samples $model_path $outname $cond_dir6 $sams 6 &
bash sample.sh $OPENAI_LOGDIR $batch_size $num_samples $model_path $outname $cond_dir7 $sams 7 


