
@REM Pendulum-v0
python train_expert.py --env_id Pendulum-v0 --num_steps 100000 --eval_interval 1000
python collect_demo.py --env_id Pendulum-v0 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Pendulum-v0/sac/seed2212/model/step100000/actor.pth
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id Pendulum-v0 --buffer buffers/Pendulum-v0/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

@REM InvertedPendulum-v2
python train_expert.py --env_id InvertedPendulum-v2 --num_steps 100000 --eval_interval 1000
python collect_demo.py --env_id InvertedPendulum-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/InvertedPendulum-v2/sac/seed2212/model/step100000/actor.pth
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo gail
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo gail --weighted
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo airl
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo airl --weighted
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo airl --state_only
python train_imitation.py --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 500 --rollout_length 2000 --algo airl --weighted --state_only


@REM Walker2d-v2
python train_expert.py --env_id Walker2d-v2 --num_steps 1000000
python collect_demo.py --env_id Walker2d-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Walker2d-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id Walker2d-v2 --buffer buffers/Walker2d-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

@REM HalfCheetah-v2
python train_expert.py --env_id HalfCheetah-v2 --num_steps 1000000
python collect_demo.py --env_id HalfCheetah-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/HalfCheetah-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id HalfCheetah-v2 --buffer buffers/HalfCheetah-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

@REM Humanoid-v2
python train_expert.py --env_id Humanoid-v2 --num_steps 1000000
python collect_demo.py --env_id Humanoid-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Humanoid-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id Humanoid-v2 --buffer buffers/Humanoid-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

@REM HumanoidStandup-v2
python train_expert.py --env_id HumanoidStandup-v2 --num_steps 1000000
python collect_demo.py --env_id HumanoidStandup-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/HumanoidStandup-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id HumanoidStandup-v2 --buffer buffers/HumanoidStandup-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only


@REM Reacher-v2
python train_expert.py --env_id Reacher-v2 --num_steps 1000000
python collect_demo.py --env_id Reacher-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Reacher-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id Reacher-v2 --buffer buffers/Reacher-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

@REM Hopper-v3
python train_expert.py --env_id Hopper-v3 --num_steps 1000000
python collect_demo.py --env_id Hopper-v3 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Hopper-v3/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id Hopper-v3 --buffer buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

@REM DisabledAnt-v0
python train_expert.py --env_id DisabledAnt-v0 --num_steps 1000000
python collect_demo.py --env_id DisabledAnt-v0 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/DisabledAnt-v0/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id DisabledAnt-v0 --buffer buffers/DisabledAnt-v0/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

@REM PointMazeLeft-v0
python train_expert.py --env_id PointMazeLeft-v0 --num_steps 1000000
python collect_demo.py --env_id PointMazeLeft-v0 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/PointMazeLeft-v0/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id PointMazeLeft-v0 --buffer buffers/PointMazeLeft-v0/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

@REM Ant-v2
python train_expert.py --env_id Ant-v2 --num_steps 1000000
python collect_demo.py --env_id Ant-v2 --buffer_size 1000000 --std 0.01 --p_rand 0.0 --weight logs/Ant-v2/sac/seed2212/model/step1000000/actor.pth
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --algo gail
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --algo gail --weighted
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --algo airl
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --algo airl --state_only
python train_imitation.py --env_id Ant-v2 --buffer buffers/Ant-v2/size1000000_std0.01_prand0.0.pth --algo airl --weighted --state_only

