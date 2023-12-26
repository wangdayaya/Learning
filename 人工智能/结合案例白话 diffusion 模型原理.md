
## 前言

本文的主要内容是使用白话介绍扩散模型 diffusion 原理，并且同时结合代码简单演示 diffusion 模型的功能，我们使用一个随机生成的“S”形图像，搭建一个“简易版的 U-Net 模型”，先加噪训练模型，然后使用训练好的模型再去噪的过程。

## 相关知识简介

首先介绍一下这个 diffusion 模型，整个模型的核心就是一个 U-Net 网络结构，它能够通过训练，在输入加噪的图片之后，学习预测出所加的噪声，这样我们在预测的时候，输入一张模糊图片，通过 U-Net 将预测的噪声去掉，不就可以尽量还原原来的图片了嘛。通俗的讲整个去噪过程分为两个步骤：

扩散模型的前向过程，主要是训练 U-Net 模型，也就是在原图的基础上加噪声，以噪声为标签，给 U-Net 输入一张带噪图片，让其预测噪声，通过不断学习，降低预测噪声和实际噪声的损失值，最后可以得到训练好的能认识噪声的 U-Net 模型。

扩散模型的反向过程，也就是还原过程，在我们训练好了上面的模型之后，给模型输入一张带噪图片，去掉 U-Net 预测的噪声，不就可以尽力还原图片了嘛。

## 数据准备

这里是直接使用 make_s_curve 来生成一个包含了 10000 个点的“S”形图像，服从标准差为 0.1 的高斯噪声点集合，因为每个点的数据都是一个三维的，我们从里面抽取第 0 维和第 2 维，形成一个二维的“S”图片，如下所示，用这个当做我们的数据集。

	import torch
	import torch.nn as nn
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn.datasets import make_s_curve
	import io
	from PIL import Image
	from tqdm import tqdm
	
	s_curve,_ = make_s_curve(10**4, noise=0.1)
	s_curve = s_curve[:, [0,2]]/10.0
	dataset = torch.Tensor(s_curve).float()
	print("数据集的大小为:", np.shape(dataset))
	
	# 绘制生成的 “S” 形图像
	data = s_curve.T
	fig,ax = plt.subplots()
	ax.scatter(*data, color='blue', edgecolor='white');
	ax.axis('off')
	plt.show()

打印结果：

	数据集的大小为: torch.Size([10000, 2])


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8a0e41f4e9454420a5ddd3ac9afc1641~tplv-k3u1fbpfcp-watermark.image?)

## 前向过程

这里展示的是前向过程，公式如下，我们可以直接使用初始图像来得到某个时间步的带噪音图像，也就是下面的 q_x 函数：

$$
q(\mathbf{x}_{t}\mid\mathbf{x}_{0}) = \mathcal{N}(\mathbf{x}_{t} ; \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0},(1-\bar{\alpha}_{t})\mathbf{I})
$$

其实写成人话就是，这些变量都是之前已经计算好的，可以直接拿来用， $\mathbf{x}_{0}$ 表示的就是最初的图片， $\mathbf{I}$ 就是由高斯分布产生的噪声：

$$
q(\mathbf{x}_{t}\mid\mathbf{x}_{0}) = \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}+\sqrt{(1-\bar{\alpha}_{t})}\mathbf{I}
$$




我们定义总共前向扩散过程有 num_steps 步，使用 torch.linspace 来初始化 num_steps 个 $\beta$ 值，取值范围区间在 start 和 end 上均匀间隔的 num_steps 个数，这样我们就得到了每个 step 时候的 $\beta$ 值 ，根据如下公式，间接地我们也同时得到了每个 step 时候的  $\alpha$ 值。

$$
\alpha_{t} = 1-\beta_{t}
$$

因为后续的在前向扩散过程中在生成某个 step 时候的噪声图是可以直接用最初的图像样本计算得到的，而这些计算需要的 $\bar{\alpha}_{t}$、 $\sqrt{\bar{\alpha}_{t}}$ 、$\sqrt{(1-\bar{\alpha}_{t})}$ 等，所以可以提前计算出来。

这里我们展示了在原始图像上加噪声的前 10 个 step 图像结果，可以看出来在到了第 5 个 step 之后基本就模糊的连他 mother 都认不出来了。

	def make_beta_schedule(n_timesteps=1000, start=1e-5, end=1e-2):
	    betas = torch.linspace(start, end, n_timesteps)
	    return betas
	# 提前计算需要的各种变量
	num_steps = 100
	betas = make_beta_schedule(n_timesteps=num_steps, start=1e-5, end=0.5e-2)
	alphas = 1-betas
	
	alphas_prod = torch.cumprod(alphas, 0)
	alphas_bar_sqrt = torch.sqrt(alphas_prod)
	one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
	
	# 前向加噪声过程
	def q_x(x_0, t):
	    noise = torch.randn_like(x_0)
	    alphas_t = alphas_bar_sqrt[t]
	    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
	    return (alphas_t * x_0 + alphas_1_m_t * noise)
	
	# 展示将图片加噪声的前 10 个结果
	num_shows = 10
	fig,axs = plt.subplots(2, 5, figsize=(10,5))
	plt.rc('text',color='black')
	for i in range(num_shows):
	    j = i//5
	    k = i%5
	    q_i = q_x(dataset, torch.tensor([i]))  
	    axs[j,k].scatter(q_i[:,0],q_i[:,1],color='red',edgecolor='white')
	    axs[j,k].set_axis_off()
	    axs[j,k].set_title('$q(\mathbf{x}_{'+str(i+1)+'})$')





![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5a222a20ade84b6396174fcbcb40395c~tplv-k3u1fbpfcp-watermark.image?)

这里定义了前向只是用最简单的方式构建了一个类似 U-Net 的模型，使用的训练集就是上面带噪声的图片，标签就是刚才我们人为加的噪声，需要预测的就是这些噪声分布，尽量减小真实噪声和预测噪声的误差。需要注意的是在进行前向传播的时候，也加入了位置编码信息。用来学习不同 step 时候的噪声。


	class MLPDiffusion(nn.Module):
	    def __init__(self, n_steps, num_units=128):
	        super(MLPDiffusion, self).__init__()
	        self.linears = nn.ModuleList(
	            [
	                nn.Linear(2, num_units),
	                nn.ReLU(),
	                nn.Linear(num_units, num_units),
	                nn.ReLU(),
	                nn.Linear(num_units, num_units),
	                nn.ReLU(),
	                nn.Linear(num_units, 2),
	            ]
	        )
	        self.step_embeddings = nn.ModuleList(
	            [
	                nn.Embedding(n_steps, num_units),
	                nn.Embedding(n_steps, num_units),
	                nn.Embedding(n_steps, num_units),
	            ]
	        )
	
	    def forward(self, x, t):
	        for idx, embedding_layer in enumerate(self.step_embeddings):
	            t_embedding = embedding_layer(t)
	            x = self.linears[2 * idx](x)
	            x += t_embedding
	            x = self.linears[2 * idx + 1](x)
	        x = self.linears[-1](x)
	        return x

这里是计算损失值的损失函数，其实就是根据上面一样的公式来计算某个 step 时候的带噪图片 x ，将这个带噪图片 x 输入模型中，让模型预测出噪声 output ，然后计算 x 和 output 两者的均方差当做损失值，我们只需要训练模型不断减少这个损失即可。


$$
q(\mathbf{x}_{t}\mid\mathbf{x}_{0}) = \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}+\sqrt{(1-\bar{\alpha}_{t})}\mathbf{I}
$$


	def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
	    batch_size = x_0.shape[0]
	    t = torch.randint(0, n_steps, size=(batch_size // 2,))
	    t = torch.cat([t, n_steps - 1 - t], dim=0)
	    t = t.unsqueeze(-1)
	    a = alphas_bar_sqrt[t]
	    aml = one_minus_alphas_bar_sqrt[t]
	    e = torch.randn_like(x_0)
	    x = x_0 * a + e * aml
	    output = model(x, t.squeeze(-1))
	    return (e - output).square().mean()
	    
## 反向过程


这里定义了反向还原过程，我们将带噪声的图片从最后的模糊模样，不断进行迭代去噪的过程，最后还原回我们的原始图片。这里需要注意的是按照原论文中规定了某个 step 的方差 $ \sigma^{2}_{t}$ =  $ \beta_{t}$ ，所以我们可以直接进行方差的计算。

这个过程应该是会比较慢的，因为要从最后一步开始，一步一步地向前进行每一步的去噪操作。p_sample_loop 最后的返回结果大小是 [101, 10000, 2] ，只有第一个元素是待去噪的原始图像输入，剩下的 100 个都是反向过程中每个 step 去噪之后的图像结果。 

	def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
	    t = torch.tensor([t])
	    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
	    eps_theta = model(x, t)
	    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
	    z = torch.randn_like(x)
	    sigma_t = betas[t].sqrt()
	    sample = mean + sigma_t * z
	    return (sample)
	
	def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
	    cur_x = torch.randn(shape)
	    x_seq = [cur_x]
	    for i in reversed(range(n_steps)):
	        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
	        x_seq.append(cur_x)
	    return x_seq	    
	    
## 训练

这里是开始训练模型的过程，为了能体现出效果，需要经历 5000 个 epoch ，每个 batch 为 128 ，使用学习率为  5*1e-4 的 Adam 当做我们的优化器。

遍历每一个 batch ，需要注意的是，每个 batch 中的样本是从 dataset 中随机抽取的 batch_size 个二维点位信息组成的图像，也就是每个样本不是一开始的 “S” 形状，而是各种由 128 点组成的图像，因为这 128 个点都是从原始“S”图中采样得到的，所以也基本都是“S”形状的（如下展示出一个样本图片） ，这主要是为了简单生成丰富我们的数据集。先通过模型前向传播计算损失，然后反向传播更新模型权重参数。

    plt.rc('text', color='blue')
    print("dataset 的 shape 是：",dataset.shape)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for idx, batch_x in enumerate(dataloader):
        print("batch 的 shape 是:",batch_x.shape)
        fig,ax = plt.subplots()
        ax.scatter(*batch_x.T, color='blue', edgecolor='white');
        ax.axis('off')
        plt.show()
        break
        
结果打印：

    dataset 的 shape 是： torch.Size([10000, 2])
    batch 的 shape 是: torch.Size([128, 2])

![S.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ac2c3927dba149a9aa0b323c207b6ef0~tplv-k3u1fbpfcp-watermark.image?)
        
        

每经历若干个 epoch ，打印一个损失值，并且使用预测好的模型进行一次反向过程，将反向过程中产生的去噪结果图片每隔 10 个 step 展示出来，从左到右的方向展示了从带噪图片输入到不断去噪过程。一共有 5000 个 epoch ，所以显示出来 5 行结果，我们可以看到在第 5 行已经初见雏形，如果 epoch 再增大，效果会更加明显。

	plt.rc('text', color='blue')
	batch_size = 128
	num_epoch = 5000
	print("dataset 的 shape 是：",dataset.shape)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
	model = MLPDiffusion(num_steps)   
	optimizer = torch.optim.Adam(model.parameters(), lr=5*1e-4)
	
	for t in tqdm(range(num_epoch)):
	    for idx, batch_x in enumerate(dataloader):
	        loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
	        optimizer.zero_grad()
	        loss.backward()
	        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
	        optimizer.step()
	
	    if (t % 1000 == 0):
	        print("第 %d 个 epoch 的损失值为 %f "%(t,loss))
	        x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)  # [101, 10000, 2]
	        fig, axs = plt.subplots(1, 10, figsize=(28, 3))
	        for i in range(1, 11):
	            cur_x = x_seq[i * 10].detach()
	            axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white');
	            axs[i - 1].set_axis_off();
	            axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 10) + '})$')    
结果打印：

	dataset 的 shape 是： torch.Size([10000, 2])




![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7389e715b62146618eb83006bfbc13e7~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9639dec94a4241048ad2b0effae968d1~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f1ff3dca7b11473fb538ffe0267281de~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f7818c2f6c1943dda0c65ff9b39d03c9~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/49e07458530a4d3e84090336a9ae93b5~tplv-k3u1fbpfcp-watermark.image?)

这里我们分别将前向过程的前 10 个 step 的加噪过程图像、和反向过程的 50 个去噪过程做成 GIF 更加形象地展示出 diffusion 模型的两个过程效果。

    imgs = []
    for i in tqdm(range(10)):
        plt.clf()
        q_i = q_x(dataset, torch.tensor([i]))
        plt.scatter(q_i[:, 0], q_i[:, 1], color='red', edgecolor='white', s=5);
        plt.axis('off');

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img = Image.open(img_buf)
        imgs.append(img)

    imgs[0].save("前向.gif",format='GIF', append_images=imgs, save_all=True, duration=300, loop=0)
   



![前向.gif](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8a0500870b444d08a2a6ffb62ab0f5d2~tplv-k3u1fbpfcp-watermark.image?)


    reverse = []
    for i in tqdm(range(0,100,2)):
        plt.clf()
        cur_x = x_seq[i].detach()
        plt.scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white', s=5);
        plt.axis('off')
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img = Image.open(img_buf)
        reverse.append(img)
    reverse[0].save("反向.gif",format='GIF',append_images=reverse, save_all=True ,duration=200, loop=0)



![反向.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/00679c1028bf4c93be3e1bc0ab3545a3~tplv-k3u1fbpfcp-watermark.image?)


## 参考
* https://github.com/cat-meowmeow/DiffusionModelDemo/blob/master/EX_S_curve/Diffusion_model_example.ipynb 
* https://huggingface.co/blog/annotated-diffusion https://zhuanlan.zhihu.com/p/572161541
* https://zhuanlan.zhihu.com/p/572161541