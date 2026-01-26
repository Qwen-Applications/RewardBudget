#!/bin/bash
# Run this script under the root directory of GARO

# bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install -U latex2sympy2_extended math_verify sseclient-py ray[default]

wandb login REDACTED_WANDB_KEY
hf auth login --token REDACTED_HF_TOKEN



echo "开始执行节点同步逻辑"
echo "RANK${RANK}" > ${PRIMUS_OUTPUT_DIR}/pre_common_ip_${PRIMUS_JOB_SUBMIT_TIMESTAMP}_${RANK}.txt

counter=`cat ${PRIMUS_OUTPUT_DIR}/pre_common_ip_${PRIMUS_JOB_SUBMIT_TIMESTAMP}_*.txt | wc -l `
while [ $counter -lt ${NNODES} ]
do
    echo "Wait for all nodes to be ready, current counter: ${counter}, all node: ${NNODES}"
    sleep 5
    counter=`cat ${PRIMUS_OUTPUT_DIR}/pre_common_ip_${PRIMUS_JOB_SUBMIT_TIMESTAMP}_*.txt | wc -l`
done

sleep 30
rm -rf ${PRIMUS_OUTPUT_DIR}/common_ip_*.txt


export RAY_TMPDIR=/tmp/ray
export RAY_PORT=6379
source recipe/GARO/scripts/ray_init.sh
init_ray
