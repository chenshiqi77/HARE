<div align="left">
<h1>
  fp8预训练
</h1>
</div>

HARE是LiteAI团队给予x万亿Tokens的开源高质量预训练数据与策略生成训练数据混合训练而成的预训练模型，模型大小1.1B，适配了transformer-engine以及ms-amp框架的fp8训练，相比bf16训练，在H800训练吞吐提升70%，取得了不错的成绩。

## 快速上手

### transformer-engine适配
* 拉取transformer-engine镜像
```shell
docker pull nvcr.io/nvidia/pytorch:23.10-py3
```
* 启动docker
``` 
docker run --gpus all -it --name fp8-mistral nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash
```
* 运行fp8训练
```shell
cd ./model
deepspeed --include='localhost:0,1,2,3,4,5,6,7' --master_port 9007 train.py ./train_args/train_config_fp8.json 
```

* 运行fp8推理
```shell
cd ./model
deepspeed --include='localhost:0,1,2,3,4,5,6,7' --master_port 9007 train.py ./train_args/train_config_fp8.json 
```

### ms-amp适配
* 拉取