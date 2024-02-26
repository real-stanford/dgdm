model_root='<directory for saving object and manipulator models>'
save_dir='<directory for saving simulation results>'
num_cpus=256     # number of cpus to use for parallel simulation

for object_idx in {0..1000}; do     # number of objects
    for ((i=0; i<1000; i+=512)) do      # number of manipulators
        python sim/sim_2d.py $model_root $i $object_idx 512 1 $save_dir $num_cpus
    done
done