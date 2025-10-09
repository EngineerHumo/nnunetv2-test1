for fold in {0..4}
do 
    # echo "nnUNetv2_train 1 3d_lowres $fold"
    #CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 1 2d $fold -num_gpus 2 
    nnUNetv2_train 1 2d $fold
done
