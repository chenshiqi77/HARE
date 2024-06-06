deepspeed --include='localhost:0,1,2,3,4,5,6,7' --master_port 9007 ./train.py ./train_args/train_config_fp8.json
