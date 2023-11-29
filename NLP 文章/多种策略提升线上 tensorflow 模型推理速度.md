# 前言

本文以最常见的模型 `Bi-LSTM-CRF` 为例，总结了在实际工作中能有效提升在 `CPU/GPU` 上的推理速度的若干方法，包括`优化模型结构`，`优化超参数`，使用 `onnx 框架`等。当然如果你有充足的 GPU ，结合以上方法提升推理速度的效果将更加震撼。

# 数据

本文使用的数据就是常见的 NER 数据，我这里使用的是 BMEO 标注方法，如下列举一个样本作为说明：

    华\B_ORG 东\M_ORG 师\M_ORG 范\M_ORG 大\M_ORG 学\E_ORG 位\O 于\O 上\B_LOC 海\E_LOC。

具体的`标注方法`和`标注规则`可以根据自己的实际业务中的实体类型进行定义，这里不做深入探讨，但是有个基本原则就是标注的实体是符合实际业务意义的内容。

# 优化模型结构

对于 `Bi-LSTM-CRF` 这一模型的具体细节，我这里默认都是知道的，所以不再赘述。我们平时在使用模型的时候有个误区觉得 LSTM 层堆叠的越多效果越好，其实不然，如果是对于入门级的 `NER` 任务，只需要`一个 Bi-LSTM` 就足够可以把实体识别出来，完全没有必要`堆叠多个 Bi-LSTM` ，这样有点杀鸡用牛刀了，而且多层的模型`参数量会激增`，这也会拖垮最终的训练和推理速度。


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f7d5299fb0804667970d304d01c6a8b1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=660&h=272&s=128681&e=png&b=fbf7f6)

对于其他的模型来说，也是同样的道理，优化模型结构，砍掉过量的层和参数，可能会取到意想不到的推理效果和速度。
    
# 优化超参数    

在我看来三个最重要的超参数就是 `batch_size` ，`hidden_size` ，`embedding_dim` ，这三个分别表示批处理样本数，隐层状态维度，嵌入纬度。这里的常见误区和模型参数量一样，会认为越大效果越好。其实不然，太大的超参数也会拖垮最终的训练和推理速度。正常在模型推理过程中，耗时基本是和这三个参数呈正相关关系。常见的参数设置可以按照以下的推荐值来进行即可：

    batch_size：32、64
    hidden_size：128、256
    embedding_dim：128、256

对于简单的 NER 任务来说，这些超参数的设置已经足够使用了，如果是比较复杂的任务，那就需要适当调大 `hidden_size` 和 `embedding_dim`，最好以 `2 的 N 次方`为值。`batch_size` 如果没有特殊业务要求，按照推荐值即可。

另外，如果你使用的是 `tensorflow2.x` 框架，可以使用 [Keras Tuner](https://tensorflow.google.cn/tutorials/keras/keras_tuner) 提到的 API ，不仅可以挑选最优的模型超参数，还能挑选最优的算法超参数。

# onnx 

`ONNX（Open Neural Network Exchange）`是一个用于表示深度学习模型的开放式`标准`。ONNX 的设计目标是使得在不同框架中训练的模型能够轻松地在其他框架中部署和运行。ONNX 支持在不同的部署环境中（例如移动设备、边缘计算、云端服务器）更加灵活地使用深度学习模型。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/21b10d4e131f41f296b667bb77e9647e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1728&h=682&s=599319&e=png&b=fcfcfc)

 ONNX 在模型部署的时候会对模型做很多优化策略，如`图结构优化、节点通信优化、量化、硬件加速、多线程和并行计算`等。`onnxruntime` 是一个对 ONNX 模型提供推理加速的 python 库，支持 CPU 和 GPU 加速，GPU 加速版本为`onnxruntime-gpu`，默认版本为 CPU 加速。安装也很简单，直接使用 `pip` 安装即可。另外安装 `tf2onnx` 需要将 `tensorflow2.x` 模型转换为 `onnx 模型`。


下面以本文中使用的模型来进行转化，需要注意的有两点，第一是要有已经训练并保存好的 h5 模型，第二是明确指定模型的输入结构，代码中的是 ` (None, config['max_len'])` ，意思是输入的 `batch_size` 可以是任意数量，输入的序列长度为 `config['max_len']` ， 具体代码如下：

```
def tensorflow2onnx():
    model = NerModel()
    model.build((None, config['max_len']))
    model.load_weights(best.h5)
    input_signature = (tf.TensorSpec((None, config['max_len']), tf.int32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature,)
    onnx.save(onnx_model, 'best.onnx')
```

保存好 onnx 模型之后，我们使用 onnx 模型进行 `CPU 推理`。只需要简单的几步即可完成推理任务， results_ort 就是推理结果 `logits` ，具体代码如下：


```
def inference():
    x_train, y_train, x_test, y_test = getData()
    sess = ort.InferenceSession(config['onnxPath'], providers=['CPUExecutionProvider'])   
    results_ort = sess.run(["output_1"], {'input': x_train})[0]
```

# 效果对比

在综合运用以上的三种，将之前的模型结构进行减小到`一层的 Bi-LSTM` ，并且将超参数进行适当的减少到都为 `256` ，然后使用` onnx 加速推理`，在 CPU 上面最终从推理速度 `278 ms` ，下降到 `29 ms` ，提升了 `9 倍`的推理速度。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e17e8f9a8b27440e8f1260d6a560a96b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=393&h=254&s=14262&e=png&b=fefefe)


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/305fd24fb7b54d85b3b3ec7b6561ed25~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=396&h=258&s=14783&e=png&b=fefefe)

如果有 GPU ，我们可以安装 `onnxruntime-gpu` （如果安装时候和 onnxruntime 有冲突，可以先卸载 onnxruntime ），然后将上面的代码改为如下即可，最终的`推理时间进一步减少了一半`:

    sess = ort.InferenceSession(config['onnxPath'], providers=['CUDAExecutionProvider'])

 

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/87b76dd763214b94b9cacc601b4381d3~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=395&h=214&s=13044&e=png&b=fefefe)

# 结论

最终我们从 `278 ms` 下降到 `15 ms` ，实现了 `18 倍`的推理提速，综上可以看出本文介绍的几种策略的综合使用确实能够加速推理速度，也说明了工业上进行模型部署优化是很有必要的。


# 参考
- https://onnxruntime.ai/docs/get-started/with-python.html
- https://tensorflow.google.cn/tutorials/keras/keras_tuner
- https://zhuanlan.zhihu.com/p/40119926
