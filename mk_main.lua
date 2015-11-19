require("io")
require("os")
require("paths")
require("torch")
dofile ("mk_p2v.lua")

config = {}
config.corpus = "test.tsv" -- input data
config.window = 5 -- (maximum) window size
config.dim = 50 -- dimensionality of word embeddings
config.alpha = 0.75 -- smooth out unigram frequencies
config.table_size = 1e8 -- table size from which to sample neg samples
config.neg_samples = 5 -- number of negative samples for each positive sample
config.minfreq = 2 --threshold for vocab frequency
config.lr = 0.025 -- initial learning rate
config.min_lr = 0.001 -- min learning rate
config.epochs = 1 -- number of epochs to train
config.gpu = 0 -- 1 = use gpu, 0 = use cpu
config.stream = 1 -- 1 = stream from hard drive 0 = copy to memory first

m = mk_p2v(config)
m:build_vocab(config.corpus)
m:build_table()

for k = 1, config.epochs do
    m.lr = config.lr -- reset learning rate at each epoch
    m:train_model(config.corpus)
end

m:dump_wordvector()
