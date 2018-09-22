#Professor forcing paper code

https://arxiv.org/pdf/1610.09038.pdf

## Progress
- [x] Generator and discriminator implemented
- [x] Char-level PTB tested
- [ ] Jupyter Notebook with results
- [ ] Word-level tests
- [ ] Sequential MNIST (also mentioned in the paper)
- [ ] Publish pre-trained model

## Requirements
- PyTorch 0.4 (older versions will not work)
- TensorboardX for tensorboard usage (optional)

## Preprocessing
No special preprocessing is required, 
just create a file with tokens separated by space.
In the original paper, the car-level language modelling is used, so in this case
you should create a single file with content like 'h e l l o _ w o r l d !'

## Training
`python train.py`
Useful command-line arguments:
- `-cuda` for GPU
- `-adversarial` for training both generator 
and discriminator (otherwise `model.generator` will
not be initialized and trained)
- `-data_path` 
- `-vocab_path`. Vocab file is created if not provided
and saved in `vocab.pt` file.
- `-save_path` Path to save the model. Model is saved
after each epoch and info about the model itself 
and its results is appended to its name.

For more parameters, consult `opts.py`

## Evaluating
`python sampler.py`

For sampling, `-checkpoint`, `data_path` 
and `vocab_path` arguments must be provided.

