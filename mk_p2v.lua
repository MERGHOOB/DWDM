require("sys")
require("nn")

local mk_p2v = torch.class("mk_p2v")
-- set the configuration....
function mk_p2v:__init(config)
	-- self.tensortype = torch.getdefaulttensortype()
	self.gpu = config.gpu -- 1 for gpu 0 for cpu
	self.stream = config.stream -- 1 from harddrvie
	self.neg_samples = config.neg_samples
	self.minfreq = config.minfreq
	self.dim = config.dim
	self.criterion = nn.BCECriterion() -- logistic loss
	-- Creates a criterion that measures
	-- Binary Cross entropy c
	-- b/w  target and output
	-- crossentropy(t,o) = -(t *log(o))+(1-t)*log(1-o);


	self.word = torch.IntTensor(1)	--?
    self.contexts = torch.IntTensor(1+self.neg_samples) 
    self.labels = torch.zeros(1+self.neg_samples); self.labels[1] = 1 -- first label is always pos sample
    self.window =config.window;
	self.lr = config.lr
	self.min_lr = config.min_lr
	self.alpha = config.alpha
	self.table_size = config.table_size
	self.vocab = {}
	self.index2word = {}
	self.word2index = {}
	self.total_count = 0
	self.tweet_count = 0
end

function mk_p2v:build_vocab(corpus)
	print ("Building vocabulary...")
	local start = sys.clock()
	local f = io.open(corpus,"r")
	local n = 1

	for line in f:lines() do
		self.tweet_count = self.tweet_count + 1
		for _,word in ipairs(self:split(line)) do
			self.total_count = self.total_count + 1
			if self.vocab[word] == nil then
				self.vocab[word] = 1
			else
				self.vocab[word] = self.vocab[word]+1
			end
		end
		n = n + 1
	end
	f:close()
	-- delete word that do not meet the min freq threshold
	--and create word indices...
	for word,count in pairs(self.vocab) do
		if count >= self.minfreq then
			self.index2word[#self.index2word+1] = word
			self.word2index[word] = #self.index2word
		else
			self.vocab[word] = nil
		end
	end
	self.vocab_size = #self.index2word
	print(self.tweet_count)
	print(string.format("%d word and %d tweets processed in %.2f second",self.total_count, self.tweet_count, sys.clock() - start))
	print(string.format("Vocab size after eliminating words occuring less than %d times: %d", self.minfreq, self.vocab_size))
	-- initialize word/context embedding now vocab size is known
	self.tweet_vecs = nn.LookupTable(self.tweet_count,self.dim) -- wordEmbedding
	self.context_vecs = nn.LookupTable(self.vocab_size,self.dim) -- contextEmbedding
	self.tweet_vecs:reset(0.25)
	self.context_vecs:reset(0.25) -- reScale N(0,1)
	self.p2v = nn.Sequential()	-- create multilayer perceptron (feed forward)
	self.p2v:add(nn.ParallelTable())
	self.p2v.modules[1]:add(self.context_vecs)
	self.p2v.modules[1]:add(self.tweet_vecs)
	self.p2v:add(nn.MM(false,true)) --dot prod and sigmoid..
	self.p2v:add(nn.Sigmoid())
    self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)
end	-- function close...

function mk_p2v:build_table()
	local start = sys.clock()
	local total_count_pow = 1
	print("Building a table of unigram frequencies....")

	for _,count in pairs(self.vocab) do
		total_count_pow = total_count_pow + count^self.alpha
	end -- smooth unigram frequencies with a factor of aplha
		
	self.table=torch.IntTensor(self.table_size);
	local word_index = 1
	local word_prob = self.vocab[self.index2word[word_index]]^self.alpha /total_count_pow

	for idx =1,self.table_size do
		self.table[idx] = word_index

		if idx  /self.table_size > word_prob then
			word_index = word_index + 1
			--print(string.format("%d %s",word_index,self.index2word[word_index]))
			if self.index2word[word_index] ~= nil then
				word_prob = word_prob + self.vocab[self.index2word[word_index]]^self.alpha/total_count_pow
			end
		end
		if word_index > self.vocab_size then
			word_index = word_index - 1
		end
	end
	print(string.format("Done in %.2f seconds",sys.clock()-start))
end

function mk_p2v:split(input, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    for str in string.gmatch(input, "([^"..sep.."]+)") do
        t[i] = str; i = i + 1
    end
    return t
end

function mk_p2v:sample_contexts(context_idx)
	self.contexts[1] = context_idx
	local  i = 0
	while i <self.neg_samples do
		neg_context = self.table[torch.random(self.table_size)]
		--print (neg_context)
		if context_idx ~=neg_context then
			self.contexts[i+2] = neg_context
			i = i + 1
		end
	end
end

function mk_p2v:normalize(m)
	m_norm = torch.zeros(m:size())
	for i=1,m:size(1) do
		m_norm[i] = m[i] / torch.norm(m[i])
	end
	return m_norm
end

function mk_p2v:train_pair(word, contexts)
    local p = self.p2v:forward({contexts, word})
    -- print ("hello")
    local loss = self.criterion:forward(p, self.labels)
    local dl_dp = self.criterion:backward(p, self.labels)
    self.p2v:zeroGradParameters()
    self.p2v:backward({contexts, word}, dl_dp)
    self.p2v:updateParameters(self.lr)
end

function mk_p2v:train_model(corpus)
    print("Training...")
    local start = sys.clock()
    local c = 0
    f = io.open(corpus, "r")
    paragraphIndex = 1
    for line in f:lines() do
        sentence = self:split(line)
        self.word[1] = paragraphIndex
        for i, word in ipairs(sentence) do
            local context = sentence[i]
            if context ~= nil then
                context_idx = self.word2index[context]
                --print(string.format("%d %d", context_idx, context))
                if context_idx ~= nil then
                    self:sample_contexts(context_idx)
                    self:train_pair(self.word, self.contexts)
                    self.lr = math.max(self.min_lr, self.lr + self.decay)
                end
            end
        end
        c = c + 1
        paragraphIndex = paragraphIndex + 1
        if c % 500 == 0 then
            print(string.format("%d tweets trained. Learning rate : %.4f", c, self.lr))
        end
    end
    print (string.format("%d paragraph vectors created", paragraphIndex))
end
function mk_p2v:dump_wordvector()
    if self.tweet_vecs_norm == nil then
        self.tweet_vecs_norm = self:normalize(self.tweet_vecs.weight:double())
    end

    local out = io.open("test_vectors.txt", "wb")
    for idx = 1, self.tweet_count do
        out:write(idx)
        for i = 1, self.dim do
            out:write(" ",string.format("%.6f",self.tweet_vecs_norm[idx][i]))
        end
        out:write("\n")
    end
    out:close()
end
