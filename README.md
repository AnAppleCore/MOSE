# OnPro
Official implementation of ICCV 2023 paper "[Online Prototype Learning for Online Continual Learning](http://arxiv.org/abs/2308.00301)".

ask the intermediate：improving online cl with multi-level supervision

## Usage
### Requirements
* python==3.8
* pytorch==1.9.0
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Training
**CIFAR-10**
```
python main.py --buffer_size 200 --mixup_p 0.6 --mixup_base_rate 0.75 --gpu_id 0
```

**CIFAR-100**
```
python main.py --dataset cifar100 --buffer_size 500 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 0
```

**TinyImageNet**
```
python main.py --dataset tiny_imagenet --buffer_size 1000 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 0
```

## Citation
If you found this code or our work useful, please cite us:

```bibtex
@inproceedings{onpro,
  title={Online Prototype Learning for Online Continual Learning},
  author={Wei, Yujie and Ye, Jiaxin and Huang, Zhizhong and Zhang, Junping and Shan, Hongming},
  booktitle={ICCV},
  year={2023}
}
```

## Temporary Experiment Log Baseline Version

- `python main.py --dataset cifar100 --buffer_size 5000 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 1`

```
total 10runs test acc results: tensor([41.2000, 42.9800, 40.7800, 41.4200, 42.7100, 42.7200, 42.8400, 40.6900,
        43.1100, 42.5300])
----------- Avg_End_Acc (42.09800012588501, 0.6864098107993939) Avg_End_Fgt (4.817000102996826, 0.6076375977591494) Avg_Acc (46.27671265091216, 0.49865995274480734) Avg_Bwtp (4.213111076354981, 0.9942734431141753) Avg_Fwt (0.0, 0.0)-----------
```

- `python main.py --dataset cifar100 --buffer_size 2000 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 1`

```
total 10runs test acc results: tensor([34.1700, 32.5800, 34.6500, 34.0800, 34.5400, 33.9200, 36.2500, 34.5600,
        34.0900, 35.1600])
----------- Avg_End_Acc (34.39999988555909, 0.6703011685675911) Avg_End_Fgt (10.66000026702881, 0.81879822770441) Avg_Acc (42.292758744330634, 0.7151425481951147) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- `python main.py --dataset cifar100 --buffer_size 1000 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 1`

```
total 10runs test acc results: tensor([27.7800, 27.2300, 27.9600, 28.2800, 28.1900, 28.4500, 27.1900, 27.1500,
        27.0300, 27.5900])
----------- Avg_End_Acc (27.685000038146974, 0.37363659030190083) Avg_End_Fgt (14.457000045776368, 0.5128670655370636) Avg_Acc (37.2788202943045, 0.626941102410949) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- `python main.py --dataset cifar100 --buffer_size 500 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 1`

```
total 10runs test acc results: tensor([20.9000, 19.4500, 19.8400, 21.2200, 20.0900, 21.7700, 20.9600, 20.6600,
        21.3200, 21.7100])
----------- Avg_End_Acc (20.792000007629394, 0.5601629930503701) Avg_End_Fgt (18.212999839782718, 0.9391936440948119) Avg_Acc (31.840424627879305, 0.801272612963133) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

## Temporary Experiment Log reduced ResNet18 Version

- `python main.py --dataset cifar100 --buffer_size 5000 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 1`

```
total 10runs test acc results: tensor([32.5200, 31.8900, 32.7200, 33.6200, 31.5600, 33.6900, 32.4400, 32.5700,
        33.5300, 32.3800])
----------- Avg_End_Acc (32.69200006961823, 0.5164108246594727) Avg_End_Fgt (3.176999912261963, 0.4157565621676871) Avg_Acc (38.06099890103227, 0.6008079651310413) Avg_Bwtp (11.942888831562467, 0.6649753785677639) Avg_Fwt (0.0, 0.0)-----------
```

## Our experiments

- SCR (1-layer head) `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

```
total 10runs test acc results: tensor([36.8800, 38.3600, 37.2200, 37.2900, 37.8600, 38.1600, 38.2700, 37.8600,   
        36.7700, 38.0100])
----------- Avg_End_Acc (37.668000049591065, 0.41609467330275507) Avg_End_Fgt (10.888000049591065, 0.5073241081038324) Avg_Acc (48.03074221282535, 0.30316397682239693) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCR (1-layer head) `python main.py --dataset cifar100 --buffer_size 2000 --lr 0.1 --gpu_id 1`

```
total 10runs test acc results: tensor([32.6800, 33.8700, 33.6600, 32.3400, 33.4000, 32.4100, 32.8800, 31.9800,
        31.3700, 32.6400])
----------- Avg_End_Acc (32.72299991607666, 0.5496109571579008) Avg_End_Fgt (17.282000465393065, 0.4637612825323395) Avg_Acc (43.926473291223004, 0.40035652856994) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD (1-layer head, ce+all_sup) `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

```
total 10runs test acc results: tensor([42.9500, 41.7400, 41.9800, 42.4900, 42.0200, 42.5200, 42.1600, 41.1400,
        41.2300, 42.9300])
----------- Avg_End_Acc (42.11599992752075, 0.4505322905708929) Avg_End_Fgt (13.797999935150148, 0.5938255514531835) Avg_Acc (51.15231848995269, 0.3033210370862692) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD (1-layer head, ce+final_sup) `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

```
total 10runs test acc results: tensor([42.0500, 40.7300, 40.8900, 40.7800, 40.5700, 40.4300, 41.8200, 40.7400,
        39.7000, 40.4800])
----------- Avg_End_Acc (40.81900016784668, 0.4839166340849684) Avg_End_Fgt (14.477999839782715, 0.348724648982239) Avg_Acc (49.104294396249074, 0.3648290480499429) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
======================================================================
```

- SCD (1-layer head, ce+all_sup+all new ce) `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

```
total 10runs test acc results: tensor([44.3900, 42.3100, 41.8400, 41.0400, 41.4000, 42.5400, 42.7800, 42.1900,
        41.4100, 42.1800])
----------- Avg_End_Acc (42.20799989700318, 0.6744179950764023) Avg_End_Fgt (14.018000164031983, 0.5611872409365933) Avg_Acc (51.12571635462746, 0.4060637296063758) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD (1-layer head, ce+all_sup+ce distill) `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

```
total 10runs test acc results: tensor([42.8200, 42.1900, 41.9000, 42.1000, 41.8400, 42.3000, 42.6200, 42.0600,
        41.9100, 42.7500])
----------- Avg_End_Acc (42.24899993896484, 0.25932062454018884) Avg_End_Fgt (13.690000114440917, 0.4513997965470356) Avg_Acc (51.391312309931195, 0.29159883143120796) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: 0.7 * 3 + 1
                - logit_distill: 0.3 * 3
                - feat_distill: 0.03 * 2
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.4100, 47.6500, 46.9500, 47.0400, 47.0600, 47.0300, 47.1200, 47.9500,
        47.0600, 48.1300])
----------- Avg_End_Acc (47.34000019073487, 0.30542380314000733) Avg_End_Fgt (13.679999732971192, 0.5225751344872727) Avg_Acc (56.071455424505565, 0.3888514153561756) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`
        
        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.7200, 47.6100, 47.4300, 46.8900, 48.0500, 46.9300, 48.0100, 49.0800,
        47.7200, 49.1500])
----------- Avg_End_Acc (47.8589998626709, 0.5491391468987429) Avg_End_Fgt (14.300000114440916, 0.5227492816537302) Avg_Acc (56.6660889132969, 0.3139496606901252) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 3`
        
        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: 0.7 * 3 + 1
                - logit_distill: 0.3 * 3
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.2700, 47.6100, 47.5500, 47.7200, 47.9900, 46.8800, 47.9500, 48.2000,
        47.6700, 47.7800])
----------- Avg_End_Acc (47.66199981689452, 0.26954195478586224) Avg_End_Fgt (13.96000011444092, 0.4018044450087984) Avg_Acc (56.342015109788804, 0.3158448210598082) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 4`
        
        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - feat_distill: 0.03 * 2
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.9000, 46.7900, 47.1300, 48.1100, 47.8200, 47.5800, 48.0000, 48.1700,
        47.2200, 48.6900])
----------- Avg_End_Acc (47.74100013732911, 0.4056837120103617) Avg_End_Fgt (13.716999740600585, 0.613708211259015) Avg_Acc (56.333355142881, 0.34643491535952614) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`
        
        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss final
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([38.8800, 39.2000, 38.7900, 38.8300, 39.5900, 39.8300, 39.3600, 39.6300,
        40.7100, 38.8400])
----------- Avg_End_Acc (39.36600006103516, 0.43323832461651895) Avg_End_Fgt (9.329000129699708, 0.6106562602794413) Avg_Acc (47.394725470300706, 0.35884981754646245) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`
        
        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss final
                - ce_loss: final
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([45.5000, 44.6300, 44.8300, 45.0900, 44.0400, 44.5700, 44.6800, 45.1300,
        45.5100, 45.9300])
----------- Avg_End_Acc (44.99099994659424, 0.3971423063179103) Avg_End_Fgt (14.888999824523927, 0.42874282604837444) Avg_Acc (53.46361572038559, 0.2650000473190247) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 3`
        
        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([42.2700, 42.5800, 42.5200, 43.4600, 43.4100, 41.9300, 42.0800, 42.7400,
        42.6000, 42.7200])
----------- Avg_End_Acc (42.630999908447265, 0.3580828951969035) Avg_End_Fgt (9.031000099182128, 0.5039129449410463) Avg_Acc (50.7526232922115, 0.3863100671765036) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 4`
        
        - 1-layer head: linear/proj
        - loss:
                - ce_loss: all
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([45.7600, 44.9700, 45.9900, 45.2200, 45.9100, 45.8700, 46.1400, 46.0600,
        46.4000, 46.4400])
----------- Avg_End_Acc (45.87599990844727, 0.3351116732012552) Avg_End_Fgt (14.540999946594237, 0.4474709493561593) Avg_Acc (54.76496666603997, 0.16585405478592125) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
        - aug: hflip + color_gray + resize_crop
        - use mem to calculate ce
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([45.3500, 45.4600, 45.6500, 46.6100, 46.5000, 45.5200, 46.4100, 45.3200,
        46.5300, 46.5900])
----------- Avg_End_Acc (45.99400001525879, 0.4094331097730342) Avg_End_Fgt (11.729000015258789, 0.42156120304931755) Avg_Acc (54.86245082299672, 0.352757937134315) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- ⭐️ SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - logit distill: 01 12 23 temp 3.0
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.9700, 48.1500, 48.6400, 48.3200, 47.6300, 48.3000, 47.7000, 48.7800,
        47.1600, 48.1900])
----------- Avg_End_Acc (48.083999977111816, 0.3481600827917081) Avg_End_Fgt (14.03700008392334, 0.48638518634088246) Avg_Acc (56.73545803310758, 0.21134686297355365) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - logit distill: 01 12 23 temp 1.0
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.0600, 48.1400, 48.6100, 47.7200, 48.3300, 47.5100, 47.6400, 48.0700,
        47.5600, 47.8600])
----------- Avg_End_Acc (47.85000003814697, 0.3221665259776423) Avg_End_Fgt (14.333999977111816, 0.4203964301044187) Avg_Acc (56.75702781530409, 0.29535961342204864) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - logit distill: 01 12 23 temp 2.0
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([48.1200, 47.7600, 48.1600, 47.1900, 48.1000, 47.3000, 47.6200, 49.5000,
        47.2500, 47.4700])
----------- Avg_End_Acc (47.847000198364256, 0.4920103876522762) Avg_End_Fgt (14.422999801635743, 0.7675352624650373) Avg_Acc (56.8295864579791, 0.2803329501290762) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`

        - 1-layer head: linear/proj
        - loss:
                - ce_loss: all
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ce

```
total 10runs test acc results: tensor([39.7500, 40.2800, 40.2500, 39.4200, 39.5300, 42.0200, 37.7200, 42.0900,
        41.6000, 39.8000])
----------- Avg_End_Acc (40.246000099182126, 0.9657299632070924) Avg_End_Fgt (19.86300003051758, 1.606896321720427) Avg_Acc (48.34417105832933, 0.6437600770997567) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 3`

        - 1-layer head: linear/proj
        - loss:
                - ce_loss: final
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ce

```
total 10runs test acc results: tensor([35.2700, 37.5500, 37.1200, 38.6300, 37.5800, 36.1300, 37.9300, 38.3900,
        35.8400, 34.4400])
----------- Avg_End_Acc (36.88799991607666, 1.0012372621515822) Avg_End_Fgt (20.808000259399414, 1.2469701385889478) Avg_Acc (44.862704071786666, 0.6691895607143519) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 4`

        - 1-layer head: linear/proj
        - loss:
                - ce_loss: all
                - logit distill: 01 12 23 temp 3.0
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ce

```
total 10runs test acc results: tensor([40.1500, 42.7400, 40.5900, 37.4100, 41.3800, 39.3900, 39.1200, 41.2300,
        41.2900, 37.0000])
----------- Avg_End_Acc (40.02999992370606, 1.3018097088814944) Avg_End_Fgt (19.808000106811523, 1.986259949101574) Avg_Acc (48.00558688587613, 0.759135141852181) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 5`

        - 1-layer head: linear/proj
        - loss:
                - ce_loss: all
                - logit distill: 01 12 23 temp 3.0
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([45.0600, 46.3500, 45.9700, 45.3900, 46.0800, 44.9600, 45.1100, 45.7500,
        45.0600, 46.1300])
----------- Avg_End_Acc (45.58599987030029, 0.37764689743481267) Avg_End_Fgt (14.69700023651123, 0.5026316769471044) Avg_Acc (54.484068689982095, 0.38552090799703825) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([41.3600, 43.4600, 43.6300, 43.7500, 42.3800, 42.4400, 41.0100, 43.1200,
        42.2200, 42.4800])
----------- Avg_End_Acc (42.58500003814697, 0.6612335486292457) Avg_End_Fgt (9.513000144958495, 0.6881821188042312) Avg_Acc (50.85177668158214, 0.3185766933173096) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - logit distill: 01 12 23 temp 3.0
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([41.6300, 43.2100, 43.6300, 43.0400, 42.5300, 42.4000, 41.8500, 42.7700,
        41.8900, 42.5400])
----------- Avg_End_Acc (42.54899993896484, 0.4577719440286847) Avg_End_Fgt (9.516000175476076, 0.6001255940762492) Avg_Acc (50.69738577373444, 0.3475763317986331) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - logit distill: 03 13 23 temp 3.0 (no detach)
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([46.9000, 48.5700, 47.9900, 48.3900, 47.3400, 47.7500, 47.0600, 48.7800,
        46.9000, 47.8500])
----------- Avg_End_Acc (47.75300003051758, 0.4944429015325908) Avg_End_Fgt (12.547000045776368, 0.8182775704575456) Avg_Acc (56.29775523883577, 0.3861223947429687) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - feat distill: 03 13 23 temp 3.0 (no detach)
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`

        - 1-layer head: linear/proj
        - loss:
                - ce_loss: all
                - distill loss:
                        - CE(pred, last_pred, 3.0)
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ce

```
total 10runs test acc results: tensor([39.3500, 42.1700, 39.4100, 39.2800, 41.3500, 39.8000, 36.9800, 42.1700,
        41.8600, 39.1900])
----------- Avg_End_Acc (40.1560000038147, 1.2055889568246654) Avg_End_Fgt (19.950999660491945, 1.673827995440183) Avg_Acc (48.44001216380558, 0.6307774938586068) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 3`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill loss:
                        - infonce(feat, feat_next, 3.0)
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.6200, 47.1200, 47.0300, 48.4700, 47.8400, 47.9000, 47.8600, 48.3100,
        47.6700, 47.9000])
----------- Avg_End_Acc (47.772000045776366, 0.32265692487074227) Avg_End_Fgt (14.33599983215332, 0.45387688726422987) Avg_Acc (56.68459992286138, 0.35912797650852707) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 4`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill loss:
                        - infonce(feat, feat_next, 2.0)
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
====================================================================================================
total 10runs test acc results: tensor([47.1600, 47.5600, 48.4200, 47.2200, 47.3700, 47.2800, 47.6300, 48.7100,
        46.7200, 48.0200])
----------- Avg_End_Acc (47.60899990081787, 0.43645818096211264) Avg_End_Fgt (14.563000183105467, 0.4781672231740184) Avg_Acc (56.645518305657404, 0.30359357341170956) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 5`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill loss:
                        - infonce(feat, feat_next, 1.0)
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.9500, 46.7600, 48.0400, 47.8000, 48.6000, 47.1700, 48.2500, 48.4700,
        47.3000, 48.6800])
----------- Avg_End_Acc (47.90200004577637, 0.463012489388432) Avg_End_Fgt (14.021999969482422, 0.47269871318508666) Avg_Acc (56.638285320811804, 0.3732118739664096) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 6`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss(proj||last_proj.detach())
                - ce_loss: all
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.1500, 47.2100, 46.9300, 48.0100, 48.3900, 47.5800, 48.1500, 47.8400,
        47.1900, 46.9200])
----------- Avg_End_Acc (47.53700008392334, 0.3805544478032922) Avg_End_Fgt (14.777999954223631, 0.2873877147707777) Avg_Acc (56.50908856655303, 0.3079349283191042) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```


- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 7`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss(proj||proj_next.detach())
                - ce_loss: all
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([47.3900, 47.3700, 47.5800, 47.2700, 47.4500, 47.6200, 46.7900, 47.5100,
        47.0900, 47.1800])
----------- Avg_End_Acc (47.32499992370605, 0.18060314840789024) Avg_End_Fgt (14.658999938964843, 0.4263894246658741) Avg_Acc (56.360910598891124, 0.2613710252177132) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
        - aug: hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm

```
total 10runs test acc results: tensor([46.5500, 46.8500, 46.9800, 48.2300, 48.0400, 46.5900, 48.0900, 47.8100,
        47.1000, 48.2000])
----------- Avg_End_Acc (47.444000015258794, 0.49508800932724073) Avg_End_Fgt (14.675000076293944, 0.467259900970078) Avg_Acc (56.43091596333943, 0.27569944183923223) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 3`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
        - aug: flip*2 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([44.2500, 43.5300, 44.8900, 44.9800, 43.4500, 43.7000, 44.7700, 43.6300,
        43.8000, 42.0600])
----------- Avg_End_Acc (43.90600009918212, 0.6251565259776395) Avg_End_Fgt (9.68899990081787, 0.4694357549700868) Avg_Acc (52.4765194601937, 0.48227433282203763) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 4`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - distill_loss: CE(apt_feat, feat, 3.0)
        - aug: flip*2 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - classifier: ncm

```
total 10runs test acc results: tensor([43.3700, 43.9200, 44.9700, 44.8300, 42.1800, 42.9000, 44.5700, 42.3500,
        43.7600, 42.9200])
----------- Avg_End_Acc (43.57699993133544, 0.7168500462537059) Avg_End_Fgt (10.089000091552734, 0.6057951269546236) Avg_Acc (52.58599444032094, 0.36300169428259865) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
        - aug: flip*2 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm

```
total 10runs test acc results: tensor([49.9100, 49.8000, 49.8400, 50.1200, 49.2600, 47.5500, 50.8100, 49.0500,
        48.6600, 47.9300])
----------- Avg_End_Acc (49.2930001449585, 0.7263530694387192) Avg_End_Fgt (14.931999702453615, 0.6562664423036847) Avg_Acc (58.61033085296268, 0.2098685805475441) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- ⭐️ ⭐️ SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 5`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill: logit 01 12 23 temp 3.0
        - aug: flip*2 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm

```
total 10runs test acc results: tensor([49.3900, 48.5800, 50.3500, 49.4800, 49.5500, 48.6500, 50.9400, 50.0500,
        48.4900, 49.1100])
----------- Avg_End_Acc (49.4590001296997, 0.5761420241807328) Avg_End_Fgt (14.449999961853027, 0.6466587415261578) Avg_Acc (58.64294597256374, 0.35491876019279467) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 6`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
        - aug: flip*8 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm

```
total 10runs test acc results: tensor([44.3700, 43.4000, 44.2400, 44.2000, 43.5200, 44.7000, 45.4100, 43.9400,
        42.9200, 44.3900])
----------- Avg_End_Acc (44.10899993896484, 0.5061870363142225) Avg_End_Fgt (9.992000122070312, 0.4478691982757209) Avg_Acc (51.013731343481275, 0.4129555533455723) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 7`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill: logit 01 12 23 temp 3.0
        - aug: flip*8 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm

```
total 10runs test acc results: tensor([44.4100, 44.9700, 43.8200, 44.7400, 43.1800, 45.4900, 45.1500, 44.9100,
        42.6800, 44.2100])
----------- Avg_End_Acc (44.35599990844727, 0.6421073996052602) Avg_End_Fgt (9.61, 0.6671467533138573) Avg_Acc (51.132652697759966, 0.36829140465396315) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 3`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
        - aug: flip*4 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm

```
total 10runs test acc results: tensor([50.0900, 49.4500, 48.5800, 49.6800, 49.4900, 49.7600, 50.3800, 49.6400,
        48.6000, 49.2400])
----------- Avg_End_Acc (49.49100002288818, 0.4107951946839067) Avg_End_Fgt (15.363000106811523, 0.37355720967615724) Avg_Acc (59.07325401885926, 0.34925095966924163) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 4`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill: logit 01 12 23 temp 3.0
        - aug: flip*4 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm

```
total 10runs test acc results: tensor([50.3700, 50.0600, 49.2400, 50.3500, 50.0800, 49.1700, 49.3000, 49.7500,
        48.8300, 48.2100])
----------- Avg_End_Acc (49.536000175476076, 0.5063078042420106) Avg_End_Fgt (15.246999626159667, 0.7384329799615558) Avg_Acc (58.86971253195263, 0.17639132086961148) Avg_Bwtp (0.0, 0.0) Avg_Fwt (0.0, 0.0)-----------
```

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 1`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill: logit 01 12 23 temp 3.0
        - aug: flip*2 + hflip + color_gray + resize_crop
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 2`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill: logit 01 12 23 temp 3.0
        - aug: flip*2 + hflip + color_gray + resize_crop
        - opt: SGD - lr 0.1 wd 5e-4 momentum 0.9
        - no attention net
        - classifier: ncm

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.001 --gpu_id 3`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill: logit 01 12 23 temp 3.0
        - aug: flip*2 + hflip + color_gray + resize_crop
        - opt: Adam - lr 0.001 wd 0.0001
        - no attention net
        - classifier: ncm

- SCD `python main.py --dataset cifar100 --buffer_size 5000 --lr 0.1 --gpu_id 4`

        - 1-layer head: linear/proj
        - loss:
                - ins_loss: sup_con_loss
                - ce_loss: all
                - distill: logit 01 12 23 temp 3.0
        - aug: flip*2 + scr_transform
        - opt: SGD 0.1
        - no attention net
        - classifier: ncm