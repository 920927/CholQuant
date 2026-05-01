export CUDA_VISIBLE_DEVICES=0

python llama.py --model model_path --method "cholquant" --wbits 3 --whitenq_n 6
python opt.py --model model_path --method "cholquant" --wbits 3 --whitenq_n 6
