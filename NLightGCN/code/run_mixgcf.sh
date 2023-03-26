python main_mgcf.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model='lgn' --epochs=500

python main.py --dataset gowalla --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 2 --pool mean --ns MixGCF --K 1 --n_negs 64 