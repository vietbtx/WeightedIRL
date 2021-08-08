export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Reacher-v2
python train_expert.py --env_id Reacher-v2 --num_steps 1000000
python collect_demo.py --env_id Reacher-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Reacher-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

# Hopper-v3
python train_expert.py --env_id Hopper-v3 --num_steps 1000000
python collect_demo.py --env_id Hopper-v3 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Hopper-v3/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

# DisabledAnt-v0
# python train_expert.py --env_id DisabledAnt-v0 --num_steps 1000000
python collect_demo.py --env_id DisabledAnt-v0 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/DisabledAnt-v0/sac/seed2212/model/step770000/actor.pth
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

# PointMazeLeft-v0
# python train_expert.py --env_id PointMazeLeft-v0 --num_steps 1000000
python collect_demo.py --env_id PointMazeLeft-v0 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/PointMazeLeft-v0/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

# Ant-v2
python train_expert.py --env_id Ant-v2 --num_steps 1000000
python collect_demo.py --env_id Ant-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Ant-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

