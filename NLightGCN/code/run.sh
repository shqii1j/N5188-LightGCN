#CUDA_VISIBLE_DEVICES=1 python main.py --decay=1e-4 --lr=0.001 --layer=2 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model='simple_n1_lgn' --epochs=500 --batch_size=2048
CUDA_VISIBLE_DEVICES=1 python main.py --decay=1e-4 --lr=0.001 --layer=2 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model='n1_lgn' --epochs=500 --batch_size=2048
