CUDA_VISIBLE_DEVICES=0 python dcrp.py --gpu 0 --label_ratio 10 --num_max 500 --imb_ratio 100 --epochs 500 --val-iteration 500 --manualSeed 0 --dataset svhn --imbalancetype long --out result_DCRP_100_10_SVHN

CUDA_VISIBLE_DEVICES=0 python dcrp.py --gpu 0 --label_ratio 20 --num_max 1000 --imb_ratio 100 --epochs 500 --val-iteration 500 --manualSeed 0 --dataset svhn --imbalancetype long --out result_DCRP_100_20_SVHN

CUDA_VISIBLE_DEVICES=0 python dcrp.py --gpu 0 --label_ratio 30 --num_max 1500 --imb_ratio 100 --epochs 500 --val-iteration 500 --manualSeed 0 --dataset svhn --imbalancetype long --out result_DCRP_100_30_SVHN

CUDA_VISIBLE_DEVICES=0 python dcrp.py --gpu 0 --label_ratio 20 --num_max 1000 --imb_ratio 50 --epochs 500 --val-iteration 500 --manualSeed 0 --dataset svhn --imbalancetype long --out result_DCRP_50_20_SVHN

