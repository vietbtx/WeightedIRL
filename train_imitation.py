import os
import argparse
import torch
from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.trainer import Trainer
from gail_airl_ppo.algo import GAIL, AIRL

def run(args):
    env = make_env(args.env_id)
    if len(args.test_env_id) == 0:
        env_test = make_env(args.env_id)
    else:
        env_test = make_env(args.test_env_id)
    buffer_exp = SerializedBuffer(path=args.buffer, device=torch.device("cuda" if args.cuda else "cpu"))

    algo_name = args.algo
    if args.weighted:
        algo_name = "w" + algo_name
    if args.algo == "gail":
        ALGO = GAIL
    elif args.algo == "airl":
        ALGO = AIRL
        if args.state_only:
            algo_name = algo_name + "_state_only"

    algo = ALGO(
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        weighted=args.weighted,
        state_only=args.state_only,
    )
    
    log_dir = os.path.join('logs', args.env_id + args.test_env_id, algo_name, f'seed{args.seed}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()
    
    if len(args.test_env_id) > 0: # Transfer Learning
        algo.train_irl = False
        algo.epoch_ppo = 100
        algo.reset()
        trainer = Trainer(
            env=env_test,
            env_test=env_test,
            algo=algo,
            log_dir=os.path.join(log_dir, 'transfer_learning'),
            num_steps=args.tl_num_steps,
            eval_interval=args.eval_interval,
            seed=args.seed
        )
        trainer.train()
        

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=2000)
    p.add_argument('--num_steps', type=int, default=1000000)
    p.add_argument('--tl_num_steps', type=int, default=1000000)
    p.add_argument('--eval_interval', type=int, default=5000)
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--test_env_id', type=str, default='')
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--weighted', action='store_true')
    p.add_argument('--state_only', action='store_true')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=2212)
    args = p.parse_args()
    run(args)
