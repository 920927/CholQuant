export CUDA_VISIBLE_DEVICES=0


python3 main.py  model_name "wikitext2" --tasks piqa,storycloze,arc_challenge,boolq,wsc,cb,rte,copa --load quantized_model_path --save "output_log_gptq"
