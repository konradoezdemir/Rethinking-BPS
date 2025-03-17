#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=30:00:00 
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH --job-name=rethinking_bps
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=yourmail@domain.com

echo "Initializing job: $SLURM_JOB_NAME with Job ID: $SLURM_JOBID"
echo "Requested resources: 1 GPUs, 50GB Memory, 48 hours runtime"
echo "Job running on partition(s): $SLURM_JOB_PARTITION"
echo "Sending job status notifications to: $SLURM_MAIL_USER"
echo "-------------------------------------"

echo "Loading Miniconda environment..."
module load devel/miniconda
echo "Activating the base conda environment..."
source activate base

echo "Switching to project-specific Conda environment: BPS"
conda activate BPS

echo "Changing directory to the Rethinking-BPS repository..."
cd .../Rethinking-BPS
echo "Current directory: $(pwd)"
echo "-------------------------------------"

# -------------------------------
echo "Starting Python script execution..."
echo "-------------------------------------"

# Define datasets, simulations, tasks, and seeds, model needs to be hardcoded yet (e.g. transformer, lstm)
declare -a DATASETS=("BPIC_2017_W")
declare -a SIMULATIONS=("real")
declare -a TASKS=("next_time") 
declare -a SEEDS=("42 123 456 789 1024 2048 4096 8192 16384 32768")

#total number of tasks for progress tracking
TOTAL_TASKS=$(( ${#DATASETS[@]} * ${#SIMULATIONS[@]} * ${#TASKS[@]} ))
COMPLETED_TASKS=0
START_TIME=$(date +%s)

progress_bar() {
    local progress=$(( (COMPLETED_TASKS * 100) / TOTAL_TASKS ))
    local filled=$(( (progress / 5) )) 
    local empty=$(( 20 - filled ))
    local elapsed_time=$(( $(date +%s) - START_TIME ))
    local avg_time=$(( (elapsed_time / (COMPLETED_TASKS+1)) ))
    local remaining_time=$(( avg_time * (TOTAL_TASKS - COMPLETED_TASKS) ))

    local eta=$(printf "%02d:%02d:%02d" $((remaining_time/3600)) $(((remaining_time%3600)/60)) $((remaining_time%60)))

    echo -ne "["
    for ((i=0; i<filled; i++)); do echo -ne "="; done
    echo -ne ">"
    for ((i=0; i<empty; i++)); do echo -ne " "; done
    echo -ne "] $progress% | Task: $COMPLETED_TASKS/$TOTAL_TASKS | ETA: $eta\r"
}

for DATASET in "${DATASETS[@]}"; do
    for SIMULATION in "${SIMULATIONS[@]}"; do
        echo -e "\nProcessing dataset: $DATASET with simulation: $SIMULATION using LSTM model..."
        for TASK in "${TASKS[@]}"; do
            echo "Executing task: $TASK..."
            if [ "$SIMULATION" == "real" ]; then
                python main.py --dataset $DATASET --cfg LSTM.yaml --sim $SIMULATION --model lstm --task $TASK --seed ${SEEDS[@]}
            else
                python main.py --dataset $DATASET --cfg LSTM.yaml --sim $SIMULATION --model lstm --task $TASK --seed 42
            fi

            ((COMPLETED_TASKS++))
            progress_bar
        done
        echo -e "\nFinished processing dataset: $DATASET with simulation: $SIMULATION"
        echo "-------------------------------------"
    done
done

COMPLETED_TASKS=$TOTAL_TASKS
progress_bar
echo -e "\nAll Python scripts have finished executing!"
echo "-------------------------------------"

# -------------------------------
echo "Cleaning up environment..."
echo "Deactivating Conda environment: BPS"
conda deactivate

# -------------------------------
echo "Sending job completion notification..."
echo "Attaching output log to email for review"
mail -s "Job $SLURM_JOBID - rethinking_bps Completed" -A output.txt yourmail@domain.com < /dev/null

# -------------------------------
echo "Job execution completed successfully!"
echo "Exiting script..."
exit 0