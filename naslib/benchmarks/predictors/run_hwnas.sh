export PYTHONPATH=${PYTHONPATH}:${HOME}/CodeGit/NASLib/

all_predictors=(
    bananas bonas gcn mlp minimlp nao seminas
    lgb ngb rf xgb
    lin_reg lasso_reg ridge_reg
    bayes_lin_reg bohamiann dngo
    gp sparse_gp
    svmr
)

all_datasets=(
    cifar10-edgegpu_energy
    cifar10-edgegpu_latency
    # cifar10-edgetpu_latency
    cifar10-eyeriss_arithmetic_intensity
    cifar10-eyeriss_energy
    cifar10-eyeriss_latency
    cifar10-fpga_energy
    cifar10-fpga_latency
    cifar10-pixel3_latency
    cifar10-raspi4_latency
    cifar100-edgegpu_energy
    cifar100-edgegpu_latency
    # cifar100-edgetpu_latency
    cifar100-eyeriss_arithmetic_intensity
    cifar100-eyeriss_energy
    cifar100-eyeriss_latency
    cifar100-fpga_energy
    cifar100-fpga_latency
    cifar100-pixel3_latency
    cifar100-raspi4_latency
    ImageNet16-120-edgegpu_energy
    ImageNet16-120-edgegpu_latency
    # ImageNet16-120-edgetpu_latency
    ImageNet16-120-eyeriss_arithmetic_intensity
    ImageNet16-120-eyeriss_energy
    ImageNet16-120-eyeriss_latency
    ImageNet16-120-fpga_energy
    ImageNet16-120-fpga_latency
    ImageNet16-120-pixel3_latency
    ImageNet16-120-raspi4_latency
)
hard_datasets=(
    ImageNet16-120-raspi4\_latency
    cifar100-pixel3\_latency
    cifar10-edgegpu\_latency
    cifar100-edgegpu\_energy
    ImageNet16-120-eyeriss\_arithmetic\_intensity
)

# --------------------------------------------

predictors=("${all_predictors[@]}")
# predictors=(lin_reg xgb)
# predictors=(ngb rf)

# datasets=("${all_datasets[@]}")
datasets=("${hard_datasets[@]}")
# datasets=(ImageNet16-120-raspi4_latency)

# --------------------------------------------

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
# base_file=NASLib/naslib
base_file=../../../../NASLib/naslib
s3_folder=/data/workspace/naslib/hw_predictors/tcml
# s3_folder=p201_im
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=hwnas

# other variables:
trials=50
end_seed=$(($start_seed + $trials - 1))
save_to_s3=false
test_size=625

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    for j in $(seq 0 $((${#datasets[@]}-1)) )
    do
        predictor=${predictors[$i]}
        dataset=${datasets[$j]}
        echo $predictor $dataset
        experiment_type=vary_train_size
        python $base_file/benchmarks/create_configs.py --predictor $predictor --experiment_type $experiment_type \
        --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
        --dataset=$dataset --config_type predictor --search_space $search_space
    done
done
