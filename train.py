import os

from argparse import ArgumentParser
from train.train_params import params_combined_cond, params_separate_cond
from train.train_config import LdmTrainConfig

def init_parser():
    parser = ArgumentParser(description='train (or resume training) a diffusion model')
    parser.add_argument(
        "--output_dir",
        default='results',
        help='directory in which to store model checkpoints and training logs'
    )
    parser.add_argument('--uniform_pitch_shift', action='store_true',
                        help="whether to apply pitch shift uniformly (as opposed to randomly)")
    parser.add_argument('--debug', action='store_true', help="whether to use debug mode")
    parser.add_argument('--load_chkpt_from', default=None, help="whether to load existing checkpoint")
    parser.add_argument('--null_cond_weight', default=0.5, help="weight parameter for null condition in classifier free guidance")
    parser.add_argument('--data_format', default="separate_melody_accompaniment", help="data format: can be 'separate_melody_accompaniment' or 'combine_melody_accompaniment'. 'separate_melody_accompaniment' means that the melody and accompaniment are in separate channels, where we generate accompaniment conditioning on chord and melody; 'combine_melody_accompaniment' means that the melody and accompaniment are combined in the same channels, where we generate melody and accompaniment conditioning on chord.")

    return parser


def args_setting_to_fn(args):
    def to_str(x: bool, char):
        return char if x else ''

    debug = to_str(args.debug, 'debug')

    return f"model-{args.data_format}-{debug}"


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    # Generate the filename based on argument settings
    fn = args_setting_to_fn(args)

    # Set the output directory
    output_dir = os.path.join(args.output_dir, fn)

    # Create the training configuration
    if args.data_format == "separate_melody_accompaniment":
        config = LdmTrainConfig(params_separate_cond, output_dir, debug_mode=args.debug, data_format="separate_melody_accompaniment", load_chkpt_from=args.load_chkpt_from)
    elif args.data_format == "combine_melody_accompaniment":
        config = LdmTrainConfig(params_combined_cond, output_dir, debug_mode=args.debug, data_format="combine_melody_accompaniment", load_chkpt_from=args.load_chkpt_from)
    else:
        raise ValueError(f"Invalid data format: {args.data_format}")

    config.train(null_rhythm_prob=args.null_cond_weight)