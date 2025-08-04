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
        default=4,
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
        '--n-epochs',
        type=int,
        default=1000,
        help='number of epochs to run (default: 1000)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor (default: 0.99)')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.2,
        help='clip ratio for epsilon-greedy action selection (default: 0.2)')
    parser.add_argument(
        '--lam',
        type=float,
        default=0.97,
        help='Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)')
    parser.add_argument(
        '--target-kl',
        type=float,
        default=0.01,
        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument(
        '--train-a-iters',
        type=int,
        default=80,
        help='Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)')
    parser.add_argument(
        '--train-v-iters',
        type=int,
        default=80,
        help='Number of gradient descent steps to take on value function per epoch.')
    parser.add_argument(
        '--n-hidden',
        type=int,
        default=64,
        help='number of hidden units in the actor and critic networks (default: 64)')

    args = parser.parse_args()

    return args
