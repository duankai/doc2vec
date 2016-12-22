# doc2vec

以下blog 来源于：http://blog.csdn.net/mrynr/article/details/52983038

通过分布式存储和分布式的词袋模型，使用层次softmax或者负采样进行深度学习
安装gensim前确保你有C语言解释器以使doc2vec的训练最优。（提升70倍）
以一个例子来初始化模型：
>>> model= Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)
将模型保存到硬盘：
>>> model.save(fname)
>>> model= Doc2Vec.load(fname) #可以下载模型来继续训练
模型同样可以从硬盘上的C格式实例化得来。
>>> model= Doc2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False) # C text format
>>> model= Doc2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True) # C binary format
  
 
class gensim.models.doc2vec.Doc2Vec(documents=None, size=300, alpha=0.025, window=8, min_count=5, max_vocab_size=None, sample=0,seed=1, workers=1, min_alpha=0.0001, dm=1, hs=1, negative=0, dbow_words=0, dm_mean=0, dm_concat=0, dm_tag_count=1, docvecs=None,docvecs_mapfile=None, comment=None, trim_rule=None, **kwargs)
Bases: gensim.models.word2vec.Word2Vec
这是用来训练的类，使用和评价http://arxiv.org/pdf/1405.4053v2.pdf中描述的神经网络。
从可迭代的文档初始化模型，每个文件都是被标记的对象，用来训练。
可迭代的文档可以是简单的一系列的被标记了的文件元素，但对于更大的语料库，考虑可直接从磁盘/网络流式传输文档的迭代。。
如果你没有提供文件，模型就不被初始化，要使用模型的话就要提供其他办法来初始化。
dm 定义了训练的算法。默认是dm=1,使用 ‘distributed memory’ (PV-DM)，否则 distributed bag of words (PV-DBOW)。
size 是特征向量的纬度。
window 是要预测的词和文档中用来预测的上下文词之间的最大距离。
alpha 是初始化的学习速率，会随着训练过程线性下降。
seed 是随机数生成器。.需要注意的是，对于一个完全明确的重复运行（fully deterministically-reproducible run），你必须同时限制模型单线程工作以消除操作系统线程调度中的有序抖动。（在python3中，解释器启动的再现要求使用PYTHONHASHSEED环境变量来控制散列随机化）
min_count 忽略总频数小于此的所有的词。
max_vocab_size 在词汇累积的时候限制内存。如果有很多独特的词多于此，则将频率低的删去。每一千万词类大概需要1G的内存，设为None以不限制（默认）。
sample 高频词被随机地降低采样的阈值。默认为0（不降低采样），较为常用的事1e-5。
workers 使用多少现成来训练模型（越快的训练需要越多核的机器）。
iter 语料库的迭代次数。从Word2Vec中继承得到的默认是5，但在已经发布的‘Paragraph Vector’中，设为10或者20是很正常的。
hs 如果为1 (默认)，分层采样将被用于模型训练（否则设为0）。
negative 如果 > 0，将使用负采样，它的值决定干扰词的个数（通常为5-20）。
dm_mean 如果为0（默认），使用上下文词向量的和；如果为1，使用均值。（仅在dm被用在非拼接模型时使用）
dm_concat 如果为1，使用上下文词向量的拼接，默认是0。注意，拼接的结果是一个更大的模型，输入的大小不再是一个词向量（采样或算术结合），而是标签和上下文中所有词结合在一起的大小。
dm_tag_count 每个文件期望的文本标签数，在使用dm_concat模式时默认为1。
dbow_words 如果设为1，训练word-vectors (in skip-gram fashion) 的同时训练 DBOW doc-vector。默认是0 (仅训练doc-vectors时更快)。
trim_rule 词汇表修建规则，用来指定某个词是否要被留下来。被删去或者作默认处理 (如果词的频数< min_count则删去)。可以设为None (将使用min_count)，或者是随时可调参 (word, count, min_count) 并返回util.RULE_DISCARD,util.RULE_KEEP ,util.RULE_DEFAULT之一。注意：这个规则只是在build_vocab()中用来修剪词汇表，而且没被保存。
 
accuracy(questions, restrict_vocab=30000, most_similar=<functionmost_similar>, case_insensitive=True)
计算模型精度。 questions 是一个文件名，其中lines是4-tuples of words, 用 ”: SECTIONNAME” lines切分。 See questions-words.txt in https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip for an example.
每部分分别输出精度的值（打印到日志并以列表形式返回），最后再加上一个总得摘要。
使用restrict_vocab 来忽视所有questions containing a word not in the first restrict_vocab words(默认30,000).如果你已经将词汇表按照频数降序排列，这将很有意义。如果 case_insensitive 为True, the first restrict_vocab words are taken first,这种情况下将执行标准化。
使用case_insensitive 在评估词汇表之前将问题和词汇表中所有的词转化为他们的大写形式 (默认为True)。在训练字符和问题词不匹配是很有用。为防止一个词的多种变体，取第一次出现的向量（同时也是最高频的，如果词汇表已经排序了的话）。
这个方法与原始的C word2veccompute-accuracy 脚本相似。
build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False)
从一系列句子中建立词汇表(可以是一次生成流a once-only generator stream)。每个句子必须是一串的unicode字符.
clear_sims()
create_binary_tree()
用根据词出现次数排好序的词汇表建立二元霍夫曼树。高频词有更短的编码，在 build_vocab()中被调用。
dbow
dm
doesnt_match(words)
列表中哪个词和别的词不匹配？
例子：
>>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
'cereal'
estimate_memory(vocab_size=None, report=None)
估计使用当前设置的模型所需的内存。
finalize_vocab(update=False)
基于最终词汇设置构建表和模型权重。
infer_vector(doc_words, alpha=0.1, min_alpha=0.0001, steps=5)
对于给定的批量（post-bulk）培训文档，推断向量。
文档应该是一连串（字）字符组成的列表。
init_sims(replace=False)
预计算L2归一化向量。
如果 replace 已经被设置，忘记原始向量，只保留归一化的值=节省大量的内存！
注意，进行替换后，您无法继续训练。模型变成高效的只读 =你可以调用 most_similar,similarity 等，但不能进行训练。
intersect_word2vec_format(fname, lockf=0.0, binary=False, encoding='utf8', unicode_errors='strict')
从给定的原始C word2vec工具格式合并输入隐藏权重矩阵，其中它与当前词汇相交。（没有字被添加到现有词汇表，但相交字采用文件的权重，并且不相交的单词被留下。）
binary 是一个布尔值，表示数据是否为二进制word2vec格式。
lockf 是要为任何导入的词矢量设置的锁定因子值;默认值0.0防止在后续训练期间向量的进一步更新。使用1.0允许进一步训练更新合并的向量。
load(*args, **kwargs)
load_word2vec_format(fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict', limit=None, datatype=<type'numpy.float32'>)
从原始C word2vec工具格式（original C word2vec-tool format）加载输入隐藏权重矩阵。
请注意，存储在文件中的信息是不完整的（二叉树丢失），因此，虽然可以查询单词相似性等，但不能继续使用以此方式加载的模型进行训练。
binary 是一个布尔值，表示数据是否为二进制word2vec格式。 norm_only 是一个布尔值，表示是否只将标准化的word2vec向量存储在存储器中。字计数从fvocab的文件名（如果有设置）读取（这是由原始C工具的-save-vocabflag生成的文件）。
如果您使用非utf 8编码为这些字训练C模型，请在encoding中指定编码。 .
unicode_errors,默认为‘strict’，是一个适合作为errors 参数传递给unicode() (Python 2.x) 或 str() (Python 3.x)函数的字符串。如果您的源文件可能包含在多字节unicode字符中间截断的字标记（正如在原始word2vec.c工具中常见的那样），“ignore”或“replace”可能有所帮助。
limit 设置从文件读取的字矢量的最大数量。默认值为None，表示读取所有。
datatype (experimental)可以将维度强制转换为非默认浮动类型（例如np.float16）以节省内存。（这种类型可能导致更慢的批量操作或与优化例程不兼容）
log_accuracy(section)
make_cum_table(power=0.75, domain=2147483647)
使用存储的词汇词计数在负抽样训练例程中绘制随机词来创建累积分布表。
要绘制单词索引，请选择一个随机整数直到表中的最大值（cum_table [-1]），然后找到整数的排序插入点（如通过bisect_left或ndarray.searchsorted()）。该插入点是绘制的索引，按比例等于该时隙处的增量。
内部调用 ‘build_vocab()’.
most_similar(positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None)
找出前N个最相似的词。积极的词有助于积极的相似性，消极的词则相反。
该方法计算给定单词投影权重向量的简单平均值与模型中每个单词的向量之间的余弦相似性。该方法对应于原始word2vec实现中的word-analogy和distance 脚本。
如果topn为False，most_similar返回相似度分数的向量。
restrict_vocab 是一个可选的整数，它限制了搜索最相似值的向量的范围。例如，restrict_vocab = 10000将只检查词汇顺序中前10000个单词向量。（如果您按频率降序排序词汇表，这将很有意义。）
例：
>>> trained_model.most_similar(positive=['woman','king'], negative=['man'])
[('queen', 0.50882536), ...]

most_similar_cosmul(positive=[], negative=[], topn=10)
使用Omer Levy和Yoav Goldberg在[4]中提出的乘法组合目标寻找前N个最相似的词。积极的词对于相似性仍然是积极地，而消极词是负面地，但对一个大距离支配计算有较小的易感性。
在常见的类比解决（analogy-solving）情况中，在两个正例和一个负例中，该方法等效于Levy和Goldberg的“3CosMul”目标（等式（4））。
附加的正或负例子分别对分子或分母做出贡献 -该方法的一个潜在敏感但未测试的扩展。（有一个正面的例子，排名将与默认most_similar中的相同）
例：
>>> trained_model.most_similar_cosmul(positive=['baghdad','england'], negative=['london'])
[(u'iraq', 0.8488819003105164), ...]
[4]
Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.
n_similarity(ws1, ws2)
计算两组字之间的余弦相似度。
例：
>>> trained_model.n_similarity(['sushi','shop'], ['japanese','restaurant'])
0.61540466561049689
 
>>> trained_model.n_similarity(['restaurant','japanese'], ['japanese','restaurant'])
1.0000000000000004
 
>>> trained_model.n_similarity(['sushi'], ['restaurant'])== trained_model.similarity('sushi','restaurant')
True

reset_from(other_model)
重复使用other_model的共享结构。
reset_weights()
save(*args, **kwargs)
将对象保存到文件。 (相同的参见 load)
fname_or_handle 是指定要保存到的文件名的字符串，或是可以写入的打开了的类似文件的对象。如果对象是文件句柄，则不执行特殊的数组处理；所有属性将被保存到同一个文件。
如果separately 为None，将自动检测正在存储的对象中的numpy / scipy.sparse数组，并将它们存储在单独的文件中。这避免了pickle内存错误，并允许有效地将mmap’ing大阵列返回到负载。
您也可以手动设置separately ，在这种情况下，它必须是存储在单独文件中的属性名称列表。在这种情况下不执行自动检查。
ignore 是一组不能序列化的属性名（文件句柄，缓存等）。在后续load()上，这些属性将设置为None。
pickle_protocol 默认为2，所以pickled对象可以在Python 2和3中导入。
save_word2vec_format(fname, fvocab=None, binary=False)
将输入隐藏权重矩阵存储为与原始C word2vec工具使用的相同的格式，以实现兼容性。
fname 是用于保存向量fvocab是用于保存词汇的可选文件binary是一个可选的布尔值，指示数据是否要以二进制word2vec格式保存（默认为False）
scale_vocab(min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None, update=False)
对min_count (放弃较不频繁的字词)和sample (控制高频字的降低采样)应用词汇设置。
调用时若 dry_run=True ，将只模拟所提供的设置，并报告保留的词汇量，有效语料库长度和估计的所需内存的大小。结果通过日志记录打印，并作为dict返回。
除非有设置keep_raw_vocab，否则在缩放完成后，删除原始词汇表以释放RAM。
scan_vocab(documents, progress_per=10000, trim_rule=None, update=False)
score(sentences, total_sentences=1000000, chunksize=100, queue_factor=2, report_delay=1)
记录一系列句子的对数概率（可以是一次性生成器流）。每个句子必须是unicode字符串的列表。这不以任何方式改变拟合模型（参见Word2Vec.train()）。
我们目前只实现分层softmax方案的分数，所以你需要以hs = 1和negative = 0运行word2vec。
注意，你应该指定total_sentences;如果你要求超过这个数量的句子的得分，将会遇到问题，但是如果设置的值太高，效率会低下。
有关如何在文档分类中使用这些分数的示例，请参阅[taddy]的文章和[deepir]的gensim演示。.
[taddy]
Taddy, Matt. Document Classification by Inversion of Distributed Language Representations, in Proceedings of the 2015 Conference of the Association of Computational Linguistics.
 
[deepir]
https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb
seeded_vector(seed_string)
创建一个“随机”向量（但由seed_string确定）
similar_by_vector(vector, topn=10, restrict_vocab=None)
通过向量找出前N个最相似的词。
如果topn为False，similar_by_vector返回相似性分数的向量。
restrict_vocab是一个可选整数，它限制了搜索最相似值的向量的范围。例如，restrict_vocab = 10000将只检查词汇顺序中前10000个单词向量。（如果您按频率降序排序词汇表，这将很有意义。）
例：
>>> trained_model.similar_by_vector([1,2])
[('survey', 0.9942699074745178), ...]

similar_by_word(word, topn=10, restrict_vocab=None)
找出前N个最相似的词。
如果topn为False，similar_by_word返回相似性分数的向量。
restrict_vocab是一个可选整数，它限制了搜索最相似值的向量的范围。例如，restrict_vocab = 10000将只检查词汇顺序中前10000个单词向量。（如果您按频率降序排序词汇表，这将很有意义。）
例：
>>> trained_model.similar_by_word('graph')
[('user', 0.9999163150787354), ...]

similarity(w1, w2)
计算两个词之间的余弦相似度。
例：
>>> trained_model.similarity('woman','man')
0.73723527
>>> trained_model.similarity('woman','woman')
1.0

sort_vocab()
排序词汇表，使最高频的单词具有最低的索引。
train(sentences, total_words=None, word_count=0, total_examples=None, queue_factor=2, report_delay=1.0)
从一系列句子更新模型的神经权重（可以是一次性生成器流）。对于Word2Vec，每个句子必须是一个unicode字符串的列表。（子类可以接受其他示例。）
为了支持从（初始）alpha到min_alpha的线性学习速率的衰减，应提供total_examples（句子数）或total_words（句子中原始单词的计数），除非句子与用于最初构建词汇表的句子相同。
update_weights()
复制所有现有权重，并重置新添加的词汇表的权重。
wmdistance(document1, document2)
计算Word Mover的两个文档之间的距离。
请注意，如果其中一个文档没有Word2Vec词汇表中存在的词语，则将返回float（'inf'）（即无穷大）。
这个方法只有当pyemd安装后才可以工作（可以通过pip安装，但需要一个C编译器）。
例：
>>> # Train word2vec model.
>>> model= Word2Vec(sentences)
>>> # Some sentences to test.
>>> sentence_obama='Obama speaks to themedia in Illinois'.lower().split()
>>> sentence_president='The president greetsthe press in Chicago'.lower().split()
>>> # Remove their stopwords.
>>> fromnltk.corpusimport stopwords
>>> stopwords= nltk.corpus.stopwords.words('english')
>>> sentence_obama= [wfor win sentence_obamaif wnotin stopwords]
>>> sentence_president= [wfor win sentence_presidentif wnotin stopwords]
>>> # Compute WMD.
>>> distance= model.wmdistance(sentence_obama,sentence_president)

class gensim.models.doc2vec.Doctag
Bases: gensim.models.doc2vec.Doctag
在初始词汇扫描期间发现的字符串文档标记。（document-vector与Vocab对象等效。）
如果所有提交的文档标签都是int，则不会使用。
The offset is only the true index into thedoctags_syn0/doctags_syn0_lockf if-and-only-if no raw-int tags were used.如果使用任何raw-int标记，则字符串Doctag向量开始于索引（max_rawint + 1），因此真索引为（rawint_index + 1+ offset）。另请参见DocvecsArray.index_to_doctag（）。
新建Doctag(offset, word_count, doc_count)实例
count(value) →integer --返回值的出现次数
doc_count：字段2的别名
index(value[, start[, stop]]) →integer --返回第一个索引值
如果值不存在，则引发ValueError
offset：字段号0的别名
repeat(word_count)
word_count：字段号1的别名
class gensim.models.doc2vec.DocvecsArray(mapfile_path=None)
Bases: gensim.utils.SaveLoad
numpy数组中在训练期间/之后的doc向量的默认存储。
作为Doc2Vec模型的'docvecs'属性，允许访问和比较文档向量。
>>> docvec= d2v_model.docvecs[99]
>>> docvec= d2v_model.docvecs['SENT_99'] # if string tag used in training
>>> sims= d2v_model.docvecs.most_similar(99)
>>> sims= d2v_model.docvecs.most_similar('SENT_99')
>>> sims= d2v_model.docvecs.most_similar(docvec)
如果在训练期间只显示纯int标签，则dict（of string tag - > index）和list（index - > string tag）保持空，节省内存。
提供mapfile_path（通过使用'docvecs_mapfile'值初始化Doc2Vec模型）将使用一对内存映射文件作为支持doctag_syn0 / doctag_syn0_lockf值的数组。
Doc2Vec模型自动使用此类，但是基于另一种持久机制（如LMDB，LevelDB或SQLite）的未来可替代实现也应该是可能的。
borrow_from(other_docvecs)
clear_sims()
doesnt_match(docs)
给定的列表中哪个文档与其他的不符？
(TODO: 接受训练集外文档的向量，如同推理一样)
estimated_lookup_memory()
标签查找的估计内存，如果使用纯int标签则为0。
index_to_doctag(i_index)
返回给定i_index的字符串键（如果可用）。否则返回raw int doctag（same int）。
indexed_doctags(doctag_tokens)
用于训练示例的返回索引和支持数组（backing-arrays）。
init_sims(replace=False)
预计算L2归一化向量。
如果设置了replace，则忘记原始向量，只保留归一化的向量=节省大量内存！
请注意，执行替换后，您无法继续训练或推断。模型变得有效只读=你可以调用most_similar，similarity 等，但不能是train 或infer_vector。
load(fname, mmap=None)
从文件加载先前保存的对象（也请参阅save）。
如果对象是使用单独存储的大型数组保存的，则可以使用mmap ='r'通过mmap（共享内存）加载这些数组。默认值：不使用mmap，将大数组作为普通对象加载。
如果正在加载的文件被压缩（'.gz'或'.bz2'），则必须设置mmap = None。如果遇到此情况，Load将引发IOError。
most_similar(positive=[], negative=[], topn=10, clip_start=0, clip_end=None, indexer=None)
找到从训中练知道的前N个最相似的docvecs。积极docs对相似性有正面影响，而消极docs为负面。
该方法计算给定文档的投影权重向量的简单平均值之间的余弦相似性。 Docs可以被指定为向量，被训练的docvecs的整数索引，或者如果文档最初通过相应的标记用字符串标记。
'clip_start'和'clip_end'允许将结果限制到底层doctag_syn0norm向量的特定连续范围。（如果选择的顺序很重要，这将会很有用，例如较低索引中的更受欢迎的标签ID）。
n_similarity(ds1, ds2)
计算来自训练集的两组docvecs之间的余弦相似性，由int index或stringtag指定。（TODO：接受训练集外文档的向量，如同推理一样）
note_doctag(key, document_no, document_length)
在初始语料库扫描期间，请记下文档标签，以便进行结构大小调整。
reset_weights(model)
save(fname_or_handle, separately=None, sep_limit=10485760, ignore=frozenset([]), pickle_protocol=2)
Save the object to file (also see load).
fname_or_handle 是指定的要保存到的文件名的字符串，或是可以写入的打开的类似文件的对象。如果对象是文件句柄，则不执行特殊的数组处理；所有属性将被保存到同一个文件。
如果 separately 为None，则自动检测正在存储的对象中的大numpy / scipy.sparse数组，并将它们存储在单独的文件中。这避免了pickle memory errors，并允许有效地将mmap’ing大阵列返回到负载。
您也可以手动单独设置，在这种情况下，它必须是存储在单独文件中的属性名称列表。在这种情况下不执行自动检查。
ignore 是一组不能序列化的属性名（文件句柄，缓存等）。在后续load()上，这些属性将设置为None。
pickle_protocol 默认为2，所以pickled对象可以在Python 2和3中导入。
similarity(d1, d2)
计算训练集中两个docvecs之间的余弦相似性，由int index或string tag指定。（TODO：接受训练集外文档的向量vectors of out-of-training-set docs，如同推理一样）
similarity_unseen_docs(model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5)
计算训练文档之间的两个后批量(post-bulk)的余弦相似性。
文档应该是一连串（字）符的列表。
trained_item(indexed_tuple)
保持对给定索引所做的任何更改（匹配由先前indexed_doctags()返回的元组）；这个实现的无操作(a no-op forthis implementation )
class gensim.models.doc2vec.LabeledSentence(*args, **kwargs)
Bases: gensim.models.doc2vec.TaggedDocument
新建 TaggedDocument(words, tags)实例
count(value) →integer --返回值的出现次数
index(value[, start[, stop]]) →integer --返回第一个索引值。如果值不存在，则引发ValueError
tags：字段号1的别名
words：字段号0的别名
class gensim.models.doc2vec.TaggedBrownCorpus(dirname)
Bases: object
迭代Brown语料库（NLTK数据的一部分）中的文档，将每个文档作为Tagged Document对象。
class gensim.models.doc2vec.TaggedDocument
Bases: gensim.models.doc2vec.TaggedDocument
单个文档，由单词（unicode字符串标记列表）和标记（标记列表）组成。标签可以是一个或多个unicode字符串标记，但典型的做法（这也将是最高效的内存）是为标签列表用唯一的整数id作为唯一的标签。
从Word2Vec替换“句子作为词列表”。
新建TaggedDocument(words, tags)实例
count(value) →integer --返回值的出现次数
index(value[, start[, stop]]) →integer --返回第一个索引值。如果值不存在，则引发ValueError。
Tags：字段号1的别名
words：字段号0的别名
class gensim.models.doc2vec.TaggedLineDocument(source)
Bases: object
简单格式：一个文档 =一行 =一个TaggedDocument对象。
预期字词必须经过预处理并以空格分隔，标签会自动从文档行号构建。
source 可以是字符串（filename）或文件对象。
例：
documents = TaggedLineDocument('myfile.txt')
或对于压缩文件：
documents = TaggedLineDocument('compressed_text.txt.bz2')
documents = TaggedLineDocument('compressed_text.txt.gz')
