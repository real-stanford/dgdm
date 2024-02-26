model_root='<directory for saving object and manipulator models>'
save_dir='<directory for saving simulation results>'
num_cpus=256

for object_idx in {0..300}; do
    for ((i=0; i<2000; i+=512)) do
        python sim/sim_3d.py $model_root $i $object_idx 512 1 $save_dir $num_cpus
    done
done