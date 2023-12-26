import os, re, json, string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tokenizers.implementations import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, TFBertModel
from keras.layers import Dense, Flatten, Layer, Input, Activation

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

max_len = 384
configuration = BertConfig()
slow_tokennizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokennizer.save_pretrained(save_path)
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)




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



def create_squad_example(raw_data):
    squad_examples = []
    for item in tqdm(raw_data['data']):
        for para in item['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                question = qa['question']
                answer_text = qa['answers'][0]['text']
                all_answers = [_['text'] for _ in qa['answers']]
                start_char_idx = qa['answers'][0]['answer_start']
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
        if not item.skip:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [dataset_dict['input_ids'], dataset_dict['token_type_ids'], dataset_dict['attention_mask']]
    y = [dataset_dict['start_token_idx'], dataset_dict['end_token_idx']]
    return x, y


def get_train_data():
    train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    train_path = keras.utils.get_file("train.json", train_data_url)
    with open(train_path) as f:
        raw_train_data = json.load(f)
    train_squad_examples = create_squad_example(raw_train_data)
    x_train, y_train = create_inputs_targets(train_squad_examples[:1000])
    print(f"{len(train_squad_examples)} training created.")
    return x_train, y_train, train_squad_examples


def get_eval_data():
    eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    eval_path = keras.utils.get_file("eval.json", eval_data_url)
    with open(eval_path) as f:
        raw_eval_data = json.load(f)
    eval_squad_examples = create_squad_example(raw_eval_data)
    x_eval, y_eval = create_inputs_targets(eval_squad_examples[:1000])
    print(f"{len(eval_squad_examples)} evaluation created.")
    return x_eval, y_eval, eval_squad_examples


def create_model():
    encoder = TFBertModel.from_pretrained("bert-base-uncased")

    input_ids = Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

    start_logits = Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = Flatten()(start_logits)

    end_logits = Dense(1, name="end_logt", use_bias=False)(embedding)
    end_logits = Flatten()(end_logits)

    start_probs = Activation(keras.activations.softmax)(start_logits)
    end_probs = Activation(keras.activations.softmax)(end_logits)

    model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[start_probs, end_probs])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model


def normalize_text(text):
    text = text.lower()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)
    text = " ".join(text.split())
    return text


class ExactMath(keras.callbacks.Callback):
    def __init__(self, x_eval, y_eval, eval_squad_examples):
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.eval_squad_examples = eval_squad_examples

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in self.eval_squad_examples if _.skip == False]
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


model = create_model()
x_train, y_train, _ = get_train_data()
x_eval, y_eval, eval_squad_examples = get_eval_data()
model.fit(x_train, y_train, epochs=1, verbose=1, batch_size=16, callbacks=[keras.callbacks.ModelCheckpoint('text_extraction', monitor='loss', save_weights_only=True), ExactMath(x_eval, y_eval, eval_squad_examples)])


# ------------------------test--------------------------
def test(x, y, eval_squad_examples):
    pred_start, pred_end = model.predict(x)
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
        print(f"预测答案{normalized_pred_ans},真实答案集合{normalized_true_ans}")
        if normalized_pred_ans in normalized_true_ans:
            count += 1
            if count == 3:
                return 
    acc = count / len(y[0])
    print(f" exact match score={acc:.2f}")


x_eval, y_eval, eval_squad_examples = get_eval_data()
model = create_model()
model.load_weights('text_extraction')
test(x_eval, y_eval, eval_squad_examples)

# Epoch 1/5
# 5384/5384 [==============================] - ETA: 0s - loss: 2.4628 - activation_loss: 1.2933 - activation_1_loss: 1.1695WARNING:absl:Found untraced functions such as embeddings_layer_call_fn, embeddings_layer_call_and_return_conditional_losses, encoder_layer_call_fn, encoder_layer_call_and_return_conditional_losses, pooler_layer_call_fn while saving (showing 5 of 420). These functions will not be directly callable after loading.
# 323/323 [==============================] - 46s 136ms/step
# epoch=1, exact match score=0.78
# 5384/5384 [==============================] - 1356s 250ms/step - loss: 2.4628 - activation_loss: 1.2933 - activation_1_loss: 1.1695
# Epoch 2/5
# 5384/5384 [==============================] - ETA: 0s - loss: 1.5804 - activation_loss: 0.8393 - activation_1_loss: 0.7411WARNING:absl:Found untraced functions such as embeddings_layer_call_fn, embeddings_layer_call_and_return_conditional_losses, encoder_layer_call_fn, encoder_layer_call_and_return_conditional_losses, pooler_layer_call_fn while saving (showing 5 of 420). These functions will not be directly callable after loading.
# 323/323 [==============================] - 44s 136ms/step
# epoch=2, exact match score=0.77
# 5384/5384 [==============================] - 1301s 242ms/step - loss: 1.5804 - activation_loss: 0.8393 - activation_1_loss: 0.7411
# Epoch 3/5
