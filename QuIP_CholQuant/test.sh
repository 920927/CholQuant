CUDA_VISIBLE_DEVICES=0 python main.py --model model_path --method "cholquant" --wbits 3 --quant "ldlq" --pre_gptqH --pre_rescale \
 --pre_proj --pre_proj_extra 1 --qfn b
 
 