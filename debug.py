from one_str_lm_iterator import OneStrLmIterator


data = dict(
    train="""trainable: whether the variable should be part of the layer's "trainable_variables" (e.g. variables,
    biases) or "non_trainable_variables" (e.g. BatchNorm mean, stddev). Note, if the current variable scope is marked
    as non-trainable then this parameter is ignored and any added variables are also marked as non-trainable. trainable
    defaults to True unless synchronization is set to ON_READ."""
)

it = OneStrLmIterator(
    data,
    num_unrollings=3,
    start_char='\n',
    verbose=True,
    shuffle=True,
    shuffle_amplitude=30,
)


for idx, b in enumerate(it.gen_batches(5)):
    print(idx, b)