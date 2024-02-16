

CUDA_VISIBLE_DEVICES=0 python dcrp.py --gpu 0 --label_ratio 20 --num_max 100 --imb_ratio 20 --epochs 500 --val-iteration 600 --manualSeed 0 --dataset cifar100 --imbalancetype long --out result_DCRP_20_20_CIFAR100

CUDA_VISIBLE_DEVICES=0 python dcrp.py --gpu 0 --label_ratio 40 --num_max 200 --imb_ratio 20 --epochs 500 --val-iteration 600 --manualSeed 0 --dataset cifar100 --imbalancetype long --out result_DCRP_40_20_CIFAR100

CUDA_VISIBLE_DEVICES=0 python dcrp.py --gpu 0 --label_ratio 40 --num_max 200 --imb_ratio 30 --epochs 500 --val-iteration 600 --manualSeed 0 --dataset cifar100 --imbalancetype long --out result_DCRP_40_30_CIFAR100

CUDA_VISIBLE_DEVICES=0 python dcrp.py --gpu 0 --label_ratio 50 --num_max 250 --imb_ratio 20 --epochs 500 --val-iteration 600 --manualSeed 0 --dataset cifar100 --imbalancetype long --out result_DCRP_20_50_CIFAR100




