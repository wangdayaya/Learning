 

## 前言

本文详细介绍了用 tensorflow-gpu 2.10 版本实现一个简单的从文本中抽取答案的过程。


## 数据准备
这里主要用于准备训练和评估 SQuAD（Standford Question Answering Dataset）数据集的 Bert 模型所需的数据和工具。

首先，通过导入相关库，包括 os、re、json、string、numpy、tensorflow、tokenizers 和 transformers，为后续处理数据和构建模型做好准备。 然后，设置了最大长度为384 ，并创建了一个 BertConfig 对象。接着从 Hugging Face 模型库中下载预训练模型 bert-base-uncased 模型的 tokenizer ，并将其保存到同一目录下的名叫 bert\_base\_uncased 文件夹中。 当下载结束之后，使用 BertWordPieceTokenizer 从已下载的文件夹中夹在 tokenizer 的词汇表从而创建分词器 tokenizer 。

剩下的部分就是从指定的 URL 下载训练和验证集，并使用 keras.utils.get\_file() 将它们保存到本地，一般存放在 “用户目录\.keras\datasets”下 ，以便后续的数据预处理和模型训练。

	import os
	import re
	import json
	import string
	import numpy as np
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras import layers
	from tokenizers import BertWordPieceTokenizer
	from transformers import BertTokenizer, TFBertModel, BertConfig
	
	max_len = 384
	configuration = BertConfig() 
	slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	save_path = "bert_base_uncased/"
	if not os.path.exists(save_path):
	    os.makedirs(save_path)
	slow_tokenizer.save_pretrained(save_path)
	
	tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
	
	train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
	train_path = keras.utils.get_file("train.json", train_data_url)
	eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
	eval_path = keras.utils.get_file("eval.json", eval_data_url)
	
打印：

	Downloading data from https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	30288272/30288272 [==============================] - 131s 4us/step
	Downloading data from https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
	4854279/4854279 [==============================] - 20s 4us/step
	
	
## 模型输入、输出处理

这里定义了一个名为 SquadExample 的类，用于表示一个 SQuAD 数据集中的问题和对应的上下文片段、答案位置等信息。

该类的构造函数 \_\_init\_\_() 接受五个参数：问题(question)、上下文(context)、答案起始字符索引(start\_char\_idx)、答案文本(answer\_text) 和所有答案列表 (all\_answers) 。

类还包括一个名为 preprocess() 的方法，用于对每个 SQuAD 样本进行预处理，首先对context 、question 和 answer 进行预处理，并计算出答案的结束位置 end\_char\_idx 。接下来，根据 start\_char\_idx 和 end\_char\_idx 在 context 的位置，构建了一个表示 context 中哪些字符属于 answer 的列表 is\_char\_in\_ans 。然后，使用 tokenizer 对 context 进行编码，得到 tokenized\_context。

接着，通过比较 answer 的字符位置和 context 中每个标记的字符位置，得到了包含答案的标记的索引列表 ans\_token\_idx 。如果 answer 未在 context 中找到，则将 skip 属性设置为 True ，并直接返回空结果。

最后，将 context 和 question 的序列拼接成输入序列 input\_ids ，并根据两个句子的不同生成了同样长度的序列 token\_type\_ids 以及与 input\_ids 同样长度的 attention\_mask 。然后对这三个序列进行了 padding 操作。

	class SquadExample:
	    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
	        self.question = question
	        self.context = context
	        self.start_char_idx = start_char_idx
	        self.answer_text = answer_text
	        self.all_answers = all_answers
	        self.skip = False
	
	    def preprocess(self):
	        context = self.context
	        question = self.question
	        answer_text = self.answer_text
	        start_char_idx = self.start_char_idx
	
	        context = " ".join(str(context).split())
	        question = " ".join(str(question).split())
	        answer = " ".join(str(answer_text).split())
	
	        end_char_idx = start_char_idx + len(answer)
	        if end_char_idx >= len(context):
	            self.skip = True
	            return
	
	        is_char_in_ans = [0] * len(context)
	        for idx in range(start_char_idx, end_char_idx):
	            is_char_in_ans[idx] = 1
	
	        tokenized_context = tokenizer.encode(context)
	
	        ans_token_idx = []
	        for idx, (start, end) in enumerate(tokenized_context.offsets):
	            if sum(is_char_in_ans[start:end]) > 0:
	                ans_token_idx.append(idx)
	
	        if len(ans_token_idx) == 0:
	            self.skip = True
	            return
	
	        start_token_idx = ans_token_idx[0]
	        end_token_idx = ans_token_idx[-1]
	
	        tokenized_question = tokenizer.encode(question)
	
	        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
	        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
	        attention_mask = [1] * len(input_ids)
	
	        padding_length = max_len - len(input_ids)
	        if padding_length > 0:   
	            input_ids = input_ids + ([0] * padding_length)
	            attention_mask = attention_mask + ([0] * padding_length)
	            token_type_ids = token_type_ids + ([0] * padding_length)
	        elif padding_length < 0:  
	            self.skip = True
	            return
	
	        self.input_ids = input_ids
	        self.token_type_ids = token_type_ids
	        self.attention_mask = attention_mask
	        self.start_token_idx = start_token_idx
	        self.end_token_idx = end_token_idx
	        self.context_token_to_char = tokenized_context.offsets

这里的两个函数用于准备数据以训练一个使用 BERT 结构的问答模型。

第一个函数 create\_squad\_examples 接受一个 JSON 文件的原始数据，将里面的每条数据都变成 SquadExample 类所定义的输入格式。

第二个函数 create\_inputs\_targets 将 SquadExample 对象列表转换为模型的输入和目标。这个函数返回两个列表，一个是模型的输入，包含了 input\_ids 、token\_type\_ids 、 attention\_mask ，另一个是模型的目标，包含了 start\_token\_idx 、end\_token\_idx。

	def create_squad_examples(raw_data):
	    squad_examples = []
	    for item in raw_data["data"]:
	        for para in item["paragraphs"]:
	            context = para["context"]
	            for qa in para["qas"]:
	                question = qa["question"]
	                answer_text = qa["answers"][0]["text"]
	                all_answers = [_["text"] for _ in qa["answers"]]
	                start_char_idx = qa["answers"][0]["answer_start"]
	                squad_eg = SquadExample(question, context, start_char_idx, answer_text, all_answers)
	                squad_eg.preprocess()
	                squad_examples.append(squad_eg)
	    return squad_examples
	
	
	def create_inputs_targets(squad_examples):
	    dataset_dict = {
	        "input_ids": [],
	        "token_type_ids": [],
	        "attention_mask": [],
	        "start_token_idx": [],
	        "end_token_idx": [],
	    }
	    for item in squad_examples:
	        if item.skip == False:
	            for key in dataset_dict:
	                dataset_dict[key].append(getattr(item, key))
	    for key in dataset_dict:
	        dataset_dict[key] = np.array(dataset_dict[key])
	
	    x = [ dataset_dict["input_ids"], dataset_dict["token_type_ids"], dataset_dict["attention_mask"], ]
	    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
	    return x, y	


这里主要读取了 SQuAD 训练集和验证集的 JSON 文件，并使用create\_squad\_examples 函数将原始数据转换为 SquadExample 对象列表。然后使用 create\_inputs\_targets 函数将这些 SquadExample 对象列表转换为模型输入和目标输出。最后输出打印了已创建的训练数据样本数和评估数据样本数。

	with open(train_path) as f:
	    raw_train_data = json.load(f)
	
	with open(eval_path) as f:
	    raw_eval_data = json.load(f)
	
	train_squad_examplesa = create_squad_examples(raw_train_data)
	x_train, y_train = create_inputs_targets(train_squad_examples)
	print(f"{len(train_squad_examples)} training points created.")
	
	eval_squad_examples = create_squad_examples(raw_eval_data)
	x_eval, y_eval = create_inputs_targets(eval_squad_examples)
	print(f"{len(eval_squad_examples)} evaluation points created.")

打印：

	87599 training points created.
	10570 evaluation points created.

## 模型搭建

这里定义了一个基于 BERT 的问答模型。在 create\_model() 函数中，首先使用 TFBertModel.from\_pretrained() 方法加载预训练的 BERT 模型。然后创建了三个输入层（input\_ids、token\_type\_ids 和 attention\_mask），每个输入层的形状都是(max\_len,) 。这些输入层用于接收模型的输入数据。

接下来使用 encoder() 方法对输入进行编码得到 embedding ，然后分别对这些向量表示进行全连接层的操作，得到一个 start\_logits 和一个 end\_logits 。接着分别对这两个向量进行扁平化操作，并将其传递到激活函数 softmax 中，得到一个 start\_probs 向量和一个 end\_probs 向量。

最后，将这三个输入层和这两个输出层传递给 keras.Model() 函数，构建出一个模型。此模型使用 SparseCategoricalCrossentropy 损失函数进行编译，并使用 Adam 优化器进行训练。

	def create_model():
	    encoder = TFBertModel.from_pretrained("bert-base-uncased")
	
	    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
	    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
	    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
	    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
	
	    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
	    start_logits = layers.Flatten()(start_logits)
	
	    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
	    end_logits = layers.Flatten()(end_logits)
	
	    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
	    end_probs = layers.Activation(keras.activations.softmax)(end_logits)
	
	    model = keras.Model( inputs=[input_ids, token_type_ids, attention_mask],  outputs=[start_probs, end_probs],)
	    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
	    optimizer = keras.optimizers.Adam(lr=5e-5)
	    model.compile(optimizer=optimizer, loss=[loss, loss])
	    return model
	    
	    
这里主要是展示了一下模型的架构，可以看到所有的参数都可以训练，并且主要调整的部分都几乎是 bert 中的参数。

	model = create_model()
	model.summary()
	
打印：

	Model: "model_1"
	__________________________________________________________________________________________________
	 Layer (type)                   Output Shape         Param #     Connected to                     
	==================================================================================================
	 input_4 (InputLayer)           [(None, 384)]        0           []                               
	                                                                                                  
	 input_6 (InputLayer)           [(None, 384)]        0           []                               
	                                                                                                  
	 input_5 (InputLayer)           [(None, 384)]        0           []                               
	                                                                                                  
	 tf_bert_model_1 (TFBertModel)  TFBaseModelOutputWi  109482240   ['input_4[0][0]',                
	                                thPoolingAndCrossAt               'input_6[0][0]',                
	                                tentions(last_hidde               'input_5[0][0]']                
	                                n_state=(None, 384,                                               
	                                 768),                                                            
	                                 pooler_output=(Non                                               
	                                e, 768),                                                          
	                                 past_key_values=No                                               
	                                ne, hidden_states=N                                               
	                                one, attentions=Non                                               
	                                e, cross_attentions                                               
	                                =None)                                                            
	                                                                                                  
	 start_logit (Dense)            (None, 384, 1)       768         ['tf_bert_model_1[0][0]']        
	                                                                                                  
	 end_logit (Dense)              (None, 384, 1)       768         ['tf_bert_model_1[0][0]']        
	                                                                                                  
	 flatten_2 (Flatten)            (None, 384)          0           ['start_logit[0][0]']            
	                                                                                                  
	 flatten_3 (Flatten)            (None, 384)          0           ['end_logit[0][0]']              
	                                                                                                  
	 activation_2 (Activation)      (None, 384)          0           ['flatten_2[0][0]']              
	                                                                                                  
	 activation_3 (Activation)      (None, 384)          0           ['flatten_3[0][0]']              
	                                                                                                  
	==================================================================================================
	Total params: 109,483,776
	Trainable params: 109,483,776
	Non-trainable params: 0

## 自定义验证回调函数

这里定义了一个回调函数 ExactMatch ， 有一个初始化方法 \_\_init\_\_ ，接收验证集的输入和目标 x\_eval 和 y\_eval  。该类还实现了 on\_epoch\_end 方法，在每个 epoch 结束时调用，计算模型的预测值，并计算精确匹配分数。

具体地，on\_epoch\_end 方法首先使用模型对 x\_eval 进行预测，得到预测的起始位置 pred\_start 和结束位置 pred\_end ，并进一步找到对应的预测答案和正确答案标准化为 normalized\_pred\_ans 和  normalized\_true\_ans ，如果前者存在于后者，则说明该样本被正确地回答，最终将精确匹配分数打印出来。

	def normalize_text(text):
	    text = text.lower()
	    exclude = set(string.punctuation)
	    text = "".join(ch for ch in text if ch not in exclude)
	    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
	    text = re.sub(regex, " ", text)
	    text = " ".join(text.split())
	    return text
	
	
	class ExactMatch(keras.callbacks.Callback):
	    def __init__(self, x_eval, y_eval):
	        self.x_eval = x_eval
	        self.y_eval = y_eval
	
	    def on_epoch_end(self, epoch, logs=None):
	        pred_start, pred_end = self.model.predict(self.x_eval)
	        count = 0
	        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
	        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
	            squad_eg = eval_examples_no_skip[idx]
	            offsets = squad_eg.context_token_to_char
	            start = np.argmax(start)
	            end = np.argmax(end)
	            if start >= len(offsets):
	                continue
	            pred_char_start = offsets[start][0]
	            if end < len(offsets):
	                pred_char_end = offsets[end][1]
	                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
	            else:
	                pred_ans = squad_eg.context[pred_char_start:]
	
	            normalized_pred_ans = normalize_text(pred_ans)
	            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
	            if normalized_pred_ans in normalized_true_ans:
	                count += 1
	        acc = count / len(self.y_eval[0])
	        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}")
## 模型训练和验证

训练模型，并使用验证集对模型的性能进行测试。这里的 epoch 只设置了 1 ，如果数值增大效果会更好。

	exact_match_callback = ExactMatch(x_eval, y_eval)
	model.fit( x_train,  y_train, epochs=1,    verbose=2,  batch_size=16, callbacks=[exact_match_callback],)

打印：


	23/323 [==============================] - 47s 139ms/step
	
	epoch=1, exact match score=0.77
	5384/5384 - 1268s - loss: 2.4677 - activation_2_loss: 1.2876 - activation_3_loss: 1.1800 - 1268s/epoch - 236ms/step
 