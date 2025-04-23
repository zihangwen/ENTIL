import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env-name',
        type=str,
        default="HalfCheetah-v5",
        help='environment to train on (default: HalfCheetah-v5)')
    parser.add_argument(
        '--num-envs',
        type=int,
        default=16,
        help='number of parallel environments (default: 16)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0,
        help='entropy term coefficient (default: 0)')
    parser.add_argument(
        '--out-dir',
        type=str,
        default="results",
        help='directory to save results (default: results)')
    parser.add_argument(
        '--T-max',
        type=int,
        default=1000,
        help='number of steps per episode (default: 1000)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='discount factor (default: 0.95)')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.1,
        help='epsilon for epsilon-greedy action selection (default: 0.1)')
    parser.add_argument(
        '--n-hidden',
        type=int,
        default=64,
        help='number of hidden units in the actor and critic networks (default: 64)')

    args = parser.parse_args()

    return args
