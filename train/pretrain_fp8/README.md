<div align="left">
<h1>
  fp8预训练
</h1>
</div>

HARE 是由中国电信股份有限公司贵州分公司 LiteAI 团队开发的预训练模型，我们使用约600B Tokens的高质量开源和策略生成的合成数据作为预训练数据。模型大小仅有1.1B，并在Open LLM Leaderboard上取得不错的成绩。LiteAI适配了transformer-engine以及ms-amp框架的fp8训练，相比bf16训练，在H800训练吞吐提升70%，取得了不错的成绩。

## 快速上手

### transformer-engine适配
#### 镜像安装
* 拉取transformer-engine镜像
```shell
docker pull nvcr.io/nvidia/pytorch:23.10-py3
```
* 启动docker
``` 
docker run --gpus all -it --name fp8-mistral nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash
```
#### 从[Transformer-Engine](https://github.com/NVIDIA/TransformerEngine)源安装
环境python 3.8.10
* 下载安装包
```
git clone --branch release_v1.7 --recursive https://github.com/NVIDIA/TransformerEngine.git
```
* 设置ds.config的精度配置
```
"bf16": {
    "enabled": true
},
```
* 适配python版本
```
cd TransformerEngine
sed -i 's/from functools import cache/from functools import lru_cache/' build_tools/utils.py
```

* 编译安装transformer-engine
```
export NVTE_FRAMEWORK=pytorch  
pip install .              
```
* 使用fp16的模型进行continue Train适配
```
1.transformers>>modeling_utils.py>>PreTrainedModel>_load _pretrained model(line 3852) 需要注意fp8的state_dict多了_extra_state属性，该属性不是tensor，需要跳过.
2.保存模型时也需要跳过_extra_state属性
```

* 运行fp8训练
```shell
cd ./model
deepspeed --include='localhost:0,1,2,3,4,5,6,7' --master_port 9007 train.py ./train_args/train_config_fp8.json 
```

* 运行fp8推理
```shell
cd ./fp8_inference_demo
python3 fp8_inference.py
```

### ms-amp适配
* 安装MPI
```
apt-get update
apt-get install mpich
```
* 拉取[ms-amp镜像](https://azure.github.io/MS-AMP/docs/user-tutorial/container-images)
```
docker pull ghcr.io/azure/msamp:${tag}
```
[ms-amp](https://github.com/Azure/MS-AMP/blob/main/dockerfile/torch2.1-cuda12.2.dockerfile)没有适配transformer-engine最新版本

### fp8效果
#### test1 
设置前提：1层Linear
```
cd test_fp8
python3 test_fp8.py
```
* 跑22个step，包含forward+backward时间结果如下：
```
shape1
==============================
use_fp8: False, bs:     8, hid_size:  2048, out_size:   512, cost_time: 1.879632ms
use_fp8: True , bs:     8, hid_size:  2048, out_size:   512, cost_time: 1.801866ms
==============================
shape2
use_fp8: False, bs:     8, hid_size:  2048, out_size:  5632, cost_time: 15.242261ms
use_fp8: True , bs:     8, hid_size:  2048, out_size:  5632, cost_time: 2.061905ms
==============================
shape3
use_fp8: False, bs:     1, hid_size:  1024, out_size:  1024, cost_time: 0.498175ms
use_fp8: True , bs:     1, hid_size:  1024, out_size:  1024, cost_time: 1.818563ms
==============================
shape4
use_fp8: False, bs:     8, hid_size:  2048, out_size:  2048, cost_time: 5.820616ms
use_fp8: True , bs:     8, hid_size:  2048, out_size:  2048, cost_time: 1.841917ms
==============================
shape5
use_fp8: False, bs:     8, hid_size:  2048, out_size:    32, cost_time: 0.535937ms
use_fp8: True , bs:     8, hid_size:  2048, out_size:    32, cost_time: 1.668860ms
==============================
```
* 结果显示使用越大的shape进行Linear计算，fp8收益越明显，在小shape的Linear使用fp8会产生负收益

#### test2
调整linear的shape，测试模型端到端性能影响：
* 调高linear的shape->HARE 2.0B模型
```
{
  "hidden_size": 8192,
  "intermediate_size": 5632,
  "num_hidden_layers": 4,
}
```
* 调小Linear的shape->HARE 0.4B模型
```
{
  "hidden_size": 1024,
  "intermediate_size": 1024,
  "num_hidden_layers": 4,
}
```
* 在两张H800上进行测试的结果
# 
| 模型  | fp8使用te.Linear | fp8不使用te.Linear  | fp16 |
|-------------------------------------|-------|-------------------|---------|
| HARE 2.0B | 55(samples/s) | 41(samples/s) | 37(samples/s)  |
| HARE 1.1B | 55(samples/s) | 57(samples/s) | 40(samples/s)  |
| HARE 0.4B | 180(samples/s)| 297(samples/s) | 265(samples/s)|
#### 结论：使用te.Linear是shape越大，性能收益越明显，shape越小会出现负收益，与test1结论一致。增加shape大小或数据大小，可以使得fp8_GEMM乘法算子内核掩盖CPU内核（量化，解量化）时间，产生端到端的收益。
#### test3
* 测试模型收敛情况
* fp8训练和fp16训练的loss均可以收敛到一致的范围
#### test4
* 测试使用te.layernorMLP，te.transformerlayer等融合算子HARE 1.1B吞吐性能
```
sed -i 's/from mistral.modeling_mistral_fp8 import MistralForCausalLM/from mistral.modeling_mistral_fp8_trans import MistralForCausalLM/' /path/HARE/train/pretrain_fp8/train.py
sh run_trainning.sh
```
#
| 模型  | fp8使用te.Linear | fp8使用te.TransformerLayer  | fp16 |
|-------------------------------------|-------|-------------------|---------|
| HARE 1.1B | 55(samples/s) | 78(samples/s) | 43(samples/s)  |
* HARE_fp8模型暂时没有对齐te.TransformerLayer设置，但是参数量相当，总体来说吞吐相比fp16提升70%以上