# Evaluate ${PRIMUS_SAVE_CHECKPOINT_DIR} using unified eval.sh script

DATA_DIR="/primus_datasets/zmy"
STEP=300
JOB_ID="primusad7f74c10a9b94a3b69982f24z"

# List of tasks to evaluate
TASKS=(
    "math500"
    "amc23"
    "aime24"
    "aime25"
    "omni_math"
    "minerva"
    "OlympiadBench_math_en_comp"
#    "OlympiadBench_math_en_cee"
    "OlympiadBench_math_zh_comp"
#    "OlympiadBench_math_zh_cee"
    "OlympiadBench_physics_en_comp"
#    "OlympiadBench_physics_en_cee"
#    "OlympiadBench_physics_zh_comp"
#    "OlympiadBench_physics_zh_cee"
)

# Evaluate each task using the unified evaluation script
for TASK in "${TASKS[@]}"; do
    echo "==================== Evaluating task: $TASK ===================="
    
    # Run the unified evaluation script
    bash recipe/GARO/scripts/evaluate/eval.sh "$TASK" "grpo" "${STEP}" "${JOB_ID}" "${DATA_DIR}" "${PRIMUS_OUTPUT_DIR}"
    
    echo "==================== Finished task: $TASK ===================="
done

echo "All evaluations completed!"
