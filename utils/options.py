import argparse
import os
import glob

def define_arguments(parser):
    parser.add_argument('--json-root', type=str, default="./data", help="")
    parser.add_argument('--feature-root', type=str, default="/scratch/pengfei4/LInEx/data", help="")
    parser.add_argument('--stream-file', type=str, default="data/MAVEN/stream_2147483647.json", help="")
    parser.add_argument('--batch-size', type=int, default=128, help="")
    parser.add_argument('--init-slots', type=int, default=13, help="")
    parser.add_argument('--patience', type=int, default=5, help="")
    parser.add_argument('--input-dim', type=int, default=2048, help="")
    parser.add_argument('--hidden-dim', type=int, default=512, help="")
    parser.add_argument('--max-slots', type=int, default=169, help="")
    parser.add_argument('--perm-id', type=int, default=0, help="")
    parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
    parser.add_argument('--gpu', type=int, default=0, help="gpu")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="")
    parser.add_argument('--decay', type=float, default=1e-2, help="")
    parser.add_argument('--kt-alpha', type=float, default=0.25, help="")
    parser.add_argument('--kt-gamma', type=float, default=0.05, help="")
    parser.add_argument('--kt-tau', type=float, default=1.0, help="")
    parser.add_argument('--kt-delta', type=float, default=0.5, help="")
    parser.add_argument('--seed', type=int, default=2147483647, help="random seed")
    parser.add_argument('--save-model', type=str, default="model", help="path to save checkpoints")
    parser.add_argument('--load-model', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--log-dir', type=str, default="./log/", help="path to save log file")
    parser.add_argument('--train-epoch', type=int, default=100, help='epochs to train')
    parser.add_argument('--test-only', action="store_true", help='is testing')
    parser.add_argument('--kt', action="store_true", help='')
    parser.add_argument('--kt2', action="store_true", help='')
    parser.add_argument('--finetune', action="store_true", help='')
    parser.add_argument('--load-first', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--skip-first', action="store_true", help='')
    parser.add_argument('--load-second', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--skip-second', action="store_true", help='')
    parser.add_argument('--balance', choices=['icarl', 'eeil', 'bic', 'none', 'fd', 'mul', 'nod'], default="none")
    parser.add_argument('--setting', choices=['classic', "new"], default="classic")

def parse_arguments():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()
    args.log = os.path.join(args.log_dir, "logfile.log")
    if (not args.test_only) and os.path.exists(args.log_dir):
        existing_logs = glob.glob(os.path.join(args.log_dir, "*"))
        for _t in existing_logs:
            os.remove(_t)
    return args
