# MOSE
Official implementation of [MOSE](links.here)

## Usage
### Requirements
* python==3.8
* pytorch==1.12.1
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

### Training and Testing
**Split CIFAR-100**

```bash
python main.py \
--dataset           cifar100 \
--buffer_size       5000 \
--method            mose \
--seed              0 \
--run_nums          5 \
--gpu_id            0
```

**Split TinyImageNet**

```bash
python main.py \
--dataset           tiny_imagenet \
--buffer_size       10000 \
--method            mose \
--seed              0 \
--run_nums          5 \
--gpu_id            0
```