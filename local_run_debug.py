from local_runner import local_run

if __name__ == "__main__":
    exp_path = "./experiments/data_generation/Go/DataGenerationlocaltry.py"
    #exp_path = "./experiments/train/policy/go_convolution_from_scratch.py"
    #local_path_bindings={"/net/scratch/people/plgantekhanke/sgfs" : "./data_processing/jgdb"})
    local_run(exp_path,False) 
 

