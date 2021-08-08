export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Pendulum-v0
# python train_expert.py --env_id Pendulum-v0 --num_steps 100000 --eval_interval 1000
python collect_demo.py --env_id Pendulum-v0 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Pendulum-v0/sac/seed2212/model/step100000/actor.pth
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

# InvertedPendulum-v2
# python train_expert.py --env_id InvertedPendulum-v2 --num_steps 100000 --eval_interval 1000
python collect_demo.py --env_id InvertedPendulum-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/InvertedPendulum-v2/sac/seed2212/model/step100000/actor.pth
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo gail
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo airl
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo wairl

# Walker2d-v2
python train_expert.py --env_id Walker2d-v2 --num_steps 1000000
python collect_demo.py --env_id Walker2d-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Walker2d-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

# HalfCheetah-v2
python train_expert.py --env_id HalfCheetah-v2 --num_steps 1000000
python collect_demo.py --env_id HalfCheetah-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/HalfCheetah-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

# Humanoid-v2
python train_expert.py --env_id Humanoid-v2 --num_steps 1000000
python collect_demo.py --env_id Humanoid-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Humanoid-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl

# HumanoidStandup-v2
python train_expert.py --env_id HumanoidStandup-v2 --num_steps 1000000
python collect_demo.py --env_id HumanoidStandup-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/HumanoidStandup-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo gail
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wgail
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo airl
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --num_steps 5000000 --eval_interval 5000 --rollout_length 2000 --algo wairl


