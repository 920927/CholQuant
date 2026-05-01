export CUDA_VISIBLE_DEVICES=0

python main.py --model model_path --cal_dataset wikitext2 --rotate --a_bits 8 --v_bits 8 --k_bits 8 --w_bits 8 --w_cli --method "cholquant"