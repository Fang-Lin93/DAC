

#gym
python main.py --env walker2d-medium-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env walker2d-medium-replay-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env walker2d-medium-expert-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10

python main.py --env hopper-medium-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1.5 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env hopper-medium-replay-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1.5 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env hopper-medium-expert-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 0.05 --rho 1.5 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10

python main.py --env halfcheetah-medium-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --q_tar lcb --rho 0 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env halfcheetah-medium-replay-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1  --q_tar lcb --rho 0 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env halfcheetah-medium-expert-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 0.1  --q_tar lcb --rho 0 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10

#antmaze
python main.py --env antmaze-umaze-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env antmaze-umaze-diverse-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb --tag Reproduce  --num_seed 8 --gpu '0, 1' --num_qs 10

python main.py --env antmaze-medium-play-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0  --rho 1 --q_tar lcb  --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env antmaze-medium-diverse-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb   --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10

python main.py --env antmaze-large-play-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0  --rho 1.1 --q_tar lcb  --tag Reproduce   --critic_lr 0.001 --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --env antmaze-large-diverse-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb   --tag Reproduce   --critic_lr 0.001 --num_seed 8 --gpu '0, 1' --num_qs 10
