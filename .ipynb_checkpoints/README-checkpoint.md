# CholQuant: Cholesky-Conditioned Fractional Whitening for LLM Quantization
**This paper proposes CholQuant, a Cholesky-conditioned fractional whitening framework for LLM quantization. The key idea is to reinterpret post-training quantization as a geometry rebalancing problem, where the Hessian-induced curvature is normalized via a Cholesky-based conditioning operator.**


## Install
1. Install all dependencies of Q2N, run:

```
conda create -n cholquant python=3.10
conda activate cholquant
pip install -r requirements.txt
```
2. If run Qwen3, please update `transformers >= 4.51.0`

## Usage
### Run & evaluate the perplexity & save fake-quantized model
1. GPTQ
```
cd GPTQ_CholQuant
bash test.sh
```

2. QuIP
```
cd QuIP_CholQuant
bash test.sh
```

3. PB-LLM
```
cd PBLLM_CholQuant/gptq_pb
bash test.sh
```

4. QuaRot
```
cd QuaRot_CholQuant/fake_quant
bash test.sh
```

### Evaluate the accuracies on downstream reasoning tasks
```
cd ZeroShot
bash test.sh
```
For running this command please follow the instruction in the repository of `lm-eval-harness`.


## Related Project
[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://github.com/IST-DASLab/gptq)

[QuIP: 2-Bit Quantization of Large Language Models with Guarantees](https://github.com/Cornell-RelaxML/QuIP)

[PB-LLM: Partially Binarized Large Language Models](https://github.com/hahnyuan/PB-LLM)

[QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://github.com/spcl/QuaRot)

[Language Model Evaluation Harness (lm-eval-harness)](https://github.com/EleutherAI/lm-evaluation-harness)