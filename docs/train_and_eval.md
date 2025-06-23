## Training
1. Train PolarBEVDet on Nuscenes with 4 GPUs:
```
bash tools/dist_train.sh projects/configs/polarbevdet/r50_704x256_24e.py 4 --work-dir work_dirs/nuscenes/polarbevdet/r50_704x256_24e
```
2. Train PolarBEVDet on Waymo with 4 GPUs:
```
bash tools/dist_train.sh projects/configs/polarbevdet_waymo/r101_832x1920_24e.py 4 --work-dir work_dirs/waymo/polarbevdet/r50_704x256_24e
```

## Evaluation
```
python tools/swap_ema_and_non_ema.py work_dirs/polarbevdet/r50_704x256_24e/iter_21096.pth

bash tools/dist_test.sh projects/configs/polarbevdet/r50_704x256_24e.py work_dirs/nuscenes/polarbevdet/r50_704x256_24e/iter_21096_ema.pth 4 --eval map
```

