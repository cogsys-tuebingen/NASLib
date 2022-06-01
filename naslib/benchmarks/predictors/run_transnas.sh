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
    class_scene
    class_object
    room_layout
    jigsaw
    segmentsemantic
)

# --------------------------------------------

# dunno: seminas
# slow, maybe hpo: bananas bonas ? dngo

predictors=("${all_predictors[@]}")
# predictors=(lin_reg)
# predictors=(ngb rf)

datasets=("${all_datasets[@]}")

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
search_space=transnas_inf

# other variables:
trials=50
end_seed=$(($start_seed + $trials - 1))
save_to_s3=false
test_size=256

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
