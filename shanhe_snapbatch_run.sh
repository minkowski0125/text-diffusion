# This is the single process file.
# 
#   RUN 
#       SNAPBATCH_PATH=/some_sharedfs snapbatch-launch -J jobname -H hostfile --env_style slurm shanhe_snapbatch_run.sh
#
# If run by snapbatch-launch, the environment variables are set.
#
#   KILL
#       pdsh -w ssh:node[x-y] "ps -ef | grep jobname | awk '{print \$2}' | xargs kill -9"

source /zhangpai21/dm/.bashrc
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_VISIBLE_DEVICES=3,7
export TOKENIZERS_PARALLELISM=false

bash -c 'echo "started at `date` on `hostname`"'
echo SLURM_NODELIST:${SLURM_NODELIST}

python main.py \
 --base configs/text-diffusion/seq2seq.yaml \
 -t --no-test \
 --devices 1  \
 --strategy "ddp_sharded" \
 --logdir "/zhangpai21/workspace/zwd/text-diffusion/logs" \
#  --precision 16 \
  --num_nodes ${SLURM_JOB_NUM_NODES} \
  -n "${SLURM_JOB_ID}.${SLURM_JOB_NAME}" \
#  --resume_from_checkpoint "/zhangpai21/checkpoints/sdpm/init/last.ckpt" \
#  --ignore-unexpected-keys \
#  --seed 1411

bash -c 'echo "ended at `date` on `hostname`"'


# --------- If env_style is "torchrun --------"
# python main.py \
#  --base configs/semantic-diffusion/decoder-dev-laion.yaml configs/exp/cc12m-shanhe-decoder.yaml \
#  -t --no-test \
#  --devices 8  \
#  --strategy "ddp_sharded" \
#  --logdir "/zhangpai21/dm/lightning_logs" \
#  --precision 16 \
# --num_nodes ${NUM_NODES} \
#   -n "${JOB_ID}.${JOB_NAME}" \
#  --resume_from_checkpoint "/zhangpai21/checkpoints/sdpm/init/last.ckpt" \