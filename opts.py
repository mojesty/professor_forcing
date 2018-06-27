"""
This file contains all options that can be passed
to either `main.py` or `sample.py`.
"""


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Generator Options
    group = parser.add_argument_group('Generator')
    group.add_argument('-embedding_size', type=int, default=32,
                       help='Embedding size')
    group.add_argument('-hidden_size', type=int, default=1024,
                       help='Hidden size')
    # group.add_argument('-vocab_size', type=int, default=-1)
    group.add_argument('-sampling_strategy', type=str,
                       default='multinomial',
                       choices=['argmax', 'multinomial'],
                       help='How to sample'
                       )
    group.add_argument('-cuda', action='store_true',
                       help='Use GPU (error will be raised if it is unavailable')
    group.add_argument('-temperature', type=float, default=3.0,
                       help='Temperature (higher is less variant) for sampling')

    group = parser.add_argument_group('Discriminator')
    group.add_argument('-d_hidden', type=int, default=512,
                       help='Hidden size of discriminator RNN')
    group.add_argument('-d_linear_size', type=int, default=512,
                       help='Size of linear layer ')
    group.add_argument('-d_dropout', type=float, default=0.2,
                       help='Dropout in dicriminator linear layers')
    # TODO: pretrained embeddings


def training_opts(parser):
    """
    Options for training.
    """
    group = parser.add_argument_group('Training')
    group.add_argument('-adversarial', action='store_true',
                       help='Use Professor Forcing or vanilla NLL')
    group.add_argument('-optional_loss', action='store_true',
                       help='Optional loss for generator')
    group.add_argument('-batch_size', type=int, default=128,
                       help='Batch size')
    group.add_argument('-bptt', type=int, default=200,
                       help='How many steps to backpropagate')
    group.add_argument('-learning_rate', type=float, default=0.0005,
                       help='Learning rate')
    group.add_argument('-clip', type=float, default=2.0,
                       help='Clip gradients if their l2 norm above threshold')
    group.add_argument('-n_epochs', type=int, default=20,
                       help='Total trainiтg epochs')
    group.add_argument('-print_every', type=int, default=20,
                       help='Print info in stdout every N batches')
    group.add_argument('-seed', type=int, default=42,
                       help='Random seed for reproducible results')
    group.add_argument('-tensorboard', action='store_true',
                       help='Use tensorboard logging')
    group.add_argument('-log_file_dir', type=str, default='./logs',
                       help="""
                       Path to tebsorboard log directory (the file name)
                       will be consistent with the checkpoint name""")
    group.add_argument('-plot_every', type=int, default=200,
                       help='Add data every N batches')


def model_io_opts(parser):
    """
    Input-Output options: checkpoint
    """
    group = parser.add_argument_group('ModelIO')

    group.add_argument('-not_save', action='store_true',
                       help='Not save the model (helpful for debug)')
    group.add_argument('-save_path', type=str, default='./model',
                       help="""Path to save the model. The path name
                       will be modified with info about model, epoch and loss
                       """)
    group.add_argument('-checkpoint', type=str, default='',
                       required=parser.description == 'sampler.py',
                       help='Path to checkpoint. If not specified, training starts from scratch')


def data_opts(parser):
    """
    Data options.
    """
    group = parser.add_argument_group('Data')

    group.add_argument('-data_path', type=str, required=True,
                       help='Path to data (plain text file)')
    group.add_argument('-vocab_path', type=str, default='',
                       help='Path to vocab .pt file generated by vocab.py')


def sample_opts(parser):
    """
    Sampling options. Note that they cannot be used simultaneously with
    train_opts.
    """

    group = parser.add_argument_group('Sampling')

    # group.add_argument('-data_path', type=str, required=True,
    #                    help='Path to data (plain text file)')
    # group.add_argument('-vocab_path', type=str, default='',
    #                    help='Path to vocab .pt file generated by vocab.py')
    group.add_argument('-batch_size', type=int, default=1,
                       help='How many sentences to sample')
    group.add_argument('-length', type=int, default=100,
                       help='Length of sampled sentence')
