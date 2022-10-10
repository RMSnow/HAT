from argparse import ArgumentParser, ArgumentTypeError

PREDICITED_TYPES = ['type', 'bar', 'pos', 'tempo',
                    'structure', 'chord', 'track', 'pitch', 'duration']
LOSSES_WEIGHTS = {'type': 5, 'bar': 5, 'pos': 1, 'tempo': 10,
                  'structure': 10, 'chord': 1, 'track': 1, 'pitch': 1, 'duration': 1}
SAMPLING_HYPARAMETER = {'type': {'t': 1.0, 'p': 0.9},
                        'bar': {'t': 1.2, 'p': 1},
                        'pos': {'t': 1.2, 'p': 1},
                        'tempo': {'t': 1.2, 'p': 0.9},
                        'structure': {'t': 1.0, 'p': 0.99},
                        'chord': {'t': 1.0, 'p': 0.99},
                        'track': {'t': 1.0, 'p': 0.9},
                        'pitch': {'t': 1.0, 'p': 0.9},
                        'duration': {'t': 2.0, 'p': 0.9}}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser()

# === Training / Generating ===
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--training', type=str2bool, default=True,
                    help='True for training, and False for generating')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--model_file', type=str)
parser.add_argument('--loss_weights', type=dict, default=LOSSES_WEIGHTS)
parser.add_argument('--sampling_hyparameters', type=dict,
                    default=SAMPLING_HYPARAMETER)
parser.add_argument('--sampling_strategy_is_nucleus',
                    type=str2bool, default=True)
parser.add_argument('--max_generated_bar', type=int, default=200)
parser.add_argument('--max_generated_song_num', type=int, default=10)
parser.add_argument('--prompt_file', type=str, default='')

# === Architecture ===
parser.add_argument('--model', type=str)
parser.add_argument('--transformer_attention_type', type=str,
                    default='full', help='linear, causal-linear, full, relative')

# *** Base ***
parser.add_argument('--max_seq_len', type=int, default=2560)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--position_dropout', type=float, default=0.1)

parser.add_argument('--song_transformer_n_layer', type=int, default=12)
parser.add_argument('--song_transformer_n_head', type=int, default=8)
parser.add_argument('--song_transformer_mlp', type=int, default=2048)
parser.add_argument('--song_transformer_dropout', type=float, default=0.1)

# *** Form ***
parser.add_argument(
    '--chord_progression_transformer_n_layer', type=int, default=6)
parser.add_argument(
    '--chord_progression_transformer_n_head', type=int, default=4)
parser.add_argument('--chord_progression_transformer_mlp',
                    type=int, default=768)
parser.add_argument('--chord_progression_transformer_dropout',
                    type=float, default=0.1)

parser.add_argument('--form_transformer_n_layer', type=int, default=12)
parser.add_argument('--form_transformer_n_head', type=int, default=8)
parser.add_argument('--form_transformer_mlp', type=int, default=1024)
parser.add_argument('--form_transformer_dropout', type=float, default=0.1)

parser.add_argument('--HAT_without_form', type=str2bool, default=False)
parser.add_argument('--HAT_without_texture', type=str2bool, default=False)

# *** Texture ***
parser.add_argument('--texture_transformer_n_layer', type=int, default=6)
parser.add_argument('--texture_transformer_n_head', type=int, default=4)
parser.add_argument('--texture_transformer_mlp', type=int, default=512)
parser.add_argument('--texture_transformer_dropout', type=float, default=0.1)

# === I/O ===
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save', type=str, default='./ckpt/debug')

# === Log ===
parser.add_argument('--verbose', type=str2bool, default=True)

# === Devices ===
parser.add_argument('--seed', type=int, default=9,
                    help='random seed')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--parallel', type=str2bool, default=False)
parser.add_argument('--fp16', type=str2bool, default=False,
                    help='use fp16 for training')
