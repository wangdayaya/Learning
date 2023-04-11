

## 前言
本文主要是介绍了使用 openai 提供的 api 来完成对开放出来的模型进行微调操作。开放的模型有 curie 、babbage、ada 等，我这里以微调 ada 举例，其他类似。

需要提前安装好 openai 所需要的各种库，我这里的库版本是 openai-0.25.0 。以及最关键过的 openai key ，这需要科学上网，请自行解决。需要注意的是微调是要花钱的，不过最开始的注册账户里默认都有 5\$ ，在开始之前到 

	https://platform.openai.com/account/usage 

这里可以查看是否有余额。另外可以去 

	https://openai.com/pricing 
wang da
查看微调不同模型的费用，对于本文的介绍的内容使用免费的 5\$ 是足够的。

## 数据准备

我们这里使用现成的数据，从网上可以直接读取使用，该数据主要有两类包含棒球和曲棍球。并且会随机打乱数据，方便后续的训练。可以看到数据的总量不大，只有 1197 条数据。

	from sklearn.datasets import fetch_20newsgroups
	import pandas as pd
	import openai
	
	categories = ['rec.sport.baseball', 'rec.sport.hockey']
	sports_dataset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories=categories)
	len_all, len_baseball, len_hockey = len(sports_dataset.data), len([e for e in sports_dataset.target if e == 0]), len([e for e in sports_dataset.target if e == 1])
	print(f"Total examples: {len_all}, Baseball examples: {len_baseball}, Hockey examples: {len_hockey}")

打印：

	Total examples: 1197, Baseball examples: 597, Hockey examples: 600

## 数据处理

为了加速我们的训练，我们这里选用打乱的训练集中的前 100 条数据来进行演示效果，因为数据多的话，时间消耗会长，而且微调的费用会和训练数据成正比增加。

这里的数据一共有两列，一列是 prompt 表示待分类的文本，一列是 completion 表示对应文本描述的标签，标签只有两类 baseball 和 hockey 。

	labels = [sports_dataset.target_names[x].split('.')[-1] for x in sports_dataset['target']]
	texts = [text.strip() for text in sports_dataset['data']]
	df = pd.DataFrame(zip(texts, labels), columns = ['prompt','completion']) 
	df = df[:100]

微调模型的输入数据需要按照规定的格式进行整理，这里使用常见的 jsonl 格式，使用 openai 库自带的工具进行处理即可得到训练集 sport2\_prepared\_train.jsonl 和验证集 sport2\_prepared\_valid.jsonl 在当前目录。

	df.to_json("sport2.jsonl", orient='records', lines=True)
	!openai tools fine_tunes.prepare_data -f sport2.jsonl -q

## 模型训练

首先将你的 openai key 设置成环境变量 OPENAI\_API\_KEY 才能执行下面的命令，该命令会使用指定的训练集和验证集进行微调的分类任务，并且会计算保留分类常见的指标，我们这里指定的模型为 ada 。



	!openai api fine_tunes.create -t "sport2_prepared_train.jsonl" -v "sport2_prepared_valid.jsonl" --compute_classification_metrics --classification_positive_class " baseball" -m ada

打印：

	Uploaded file from sport2_prepared_train.jsonl: file-wx9c3lYQB6Z4pWrrCqBabWUh
	Uploaded file from sport2_prepared_valid.jsonl: file-aujZlpbhXZnevKzJNjF06q85
	Created fine-tune: ft-aEHXhd8q9dfG8MOKt43ph7wk
	Streaming events until fine-tuning is complete...
	[2023-03-28 09:57:12] Created fine-tune: ft-aEHXhd8q9dfG8MOKt43ph7wk
	[2023-03-28 09:59:16] Fine-tune costs $0.06
	[2023-03-28 09:59:16] Fine-tune enqueued. Queue number: 2
	[2023-03-28 09:59:32] Fine-tune is in the queue. Queue number: 1
	(Ctrl-C will interrupt the stream, but not cancel the fine-tune)
	[2023-03-28 09:57:12] Created fine-tune: ft-aEHXhd8q9dfG8MOKt43ph7wk
	
	Stream interrupted (client disconnected).
	To resume the stream, run:
	
	  openai api fine_tunes.follow -i ft-aEHXhd8q9dfG8MOKt43ph7wk
	  
从打印信息中我们能看到此次训练的花费，以及当前的排队情况，这个训练过程是在 openai 的服务器上进行的，有时候长时间因为排队没有响应会自己断开数据流的传输，我们如果想要继续查看任务情况，只需要找到打印出来的唯一任务编码，执行下面的命令，我的远程服务器上的训练任务编码是 ft-aEHXhd8q9dfG8MOKt43ph7wk ，其实上面的打印信息中都有相应的提示。

	openai api fine_tunes.follow -i ft-aEHXhd8q9dfG8MOKt43ph7wk
	

	[2023-03-28 09:57:12] Created fine-tune: ft-aEHXhd8q9dfG8MOKt43ph7wk
	[2023-03-28 09:59:16] Fine-tune costs $0.06
	[2023-03-28 09:59:16] Fine-tune enqueued. Queue number: 2
	[2023-03-28 09:59:32] Fine-tune is in the queue. Queue number: 1
	[2023-03-28 10:12:20] Fine-tune is in the queue. Queue number: 0
	[2023-03-28 10:13:54] Fine-tune started
	[2023-03-28 10:14:22] Completed epoch 1/4
	[2023-03-28 10:14:37] Completed epoch 2/4
	[2023-03-28 10:14:50] Completed epoch 3/4
	[2023-03-28 10:15:03] Completed epoch 4/4
	[2023-03-28 10:15:26] Uploaded model: ada:ft-personal-2023-03-28-02-15-26
	[2023-03-28 10:15:27] Uploaded result file: file-YZ2VNHkFnAJAhBeTKJ2AxfLK
	[2023-03-28 10:15:27] Fine-tune succeeded

从打印信息中我们可以看到微调的结果模型叫 ada:ft-personal-2023-03-28-02-15-26 ，这个可以在 https://platform.openai.com/playground 里的模型选择栏中看到自己微调后的模型。

## 训练信息打印

我们通过任务编码可以获取该任务训练的各种信息，比如随着 epoch 变化的 loss 、acc 等信息。可以看出在我们的训练集上训练的分类准确率为 100% 。

	!openai api fine_tunes.results -i ft-aEHXhd8q9dfG8MOKt43ph7wk > result.csv
	results = pd.read_csv('result.csv')
	results[results['classification/accuracy'].notnull()].tail(1)
	
打印信息：

		step	elapsed_tokens	elapsed_examples	training_loss	training_sequence_accuracy	training_token_accuracy	validation_loss	validation_sequence_accuracy	validation_token_accuracy	classification/accuracy	classification/precision	classification/recall	classification/auroc	classification/auprc	classification/f1.0
	316	317	143557	317	0.02417	1.0	1.0	NaN	NaN	NaN	1.0	1.0	1.0	1.0	1.0	1.0
	
## 模型测试

我们随机挑选验证集中的一条文本，使用微调后的模型进行测试，打印出来的分类标签是正确的。

	test = pd.read_json('sport2_prepared_valid.jsonl', lines=True)
	res = openai.Completion.create(model= 'ada:ft-personal-2023-03-28-02-15-26', prompt=test['prompt'][0] + '\n\n###\n\n', max_tokens=1, temperature=0)
	res['choices'][0]['text']

打印：

	' hockey'

另外我们的微调分类器是非常通用的，不仅在我们使用的训练集和验证集上游泳，它也能用来预测推文。

	sample_hockey_tweet = """Thank you to the 
	@Canes
	 and all you amazing Caniacs that have been so supportive! You guys are some of the best fans in the NHL without a doubt! Really excited to start this new chapter in my career with the 
	@DetroitRedWings
	 !!"""
	res = openai.Completion.create(model='ada:ft-personal-2023-03-28-02-15-26', prompt=sample_hockey_tweet + '\n\n###\n\n', max_tokens=1, temperature=0, logprobs=2)
	res['choices'][0]['text']

打印：

	' baseball'
	
## 总结

其实使用 openai 的微调 api 只需要四步：

* 准备环境和 key
* 准备规定格式的数据
* 训练模型
* 模型推理

是不是很简单！