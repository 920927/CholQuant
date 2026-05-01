export CUDA_VISIBLE_DEVICES=0


python run.py --model model_path --dataset wikitext2 --low_quant_method xnor --low_frac 0.9 --high_bit 8 --salient_metric hessian --method 'cholquant'
