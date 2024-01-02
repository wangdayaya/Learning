import keras_nlp, time
import tensorflow as tf
import tensorflow_datasets as tfds
import os, json
tf.keras.mixed_precision.set_global_policy("mixed_float16")

for gpu in tf.config.experimental.list_physical_devices(device_type='gpu'):
    tf.config.experimental.set_memory_growth(gpu, True)

# 我爱中国！
# GPT-2 生成的文本如下：
# 我爱中国！拳探品的经和没有那与罗没有那格拳探品的品没有那格拳探品的品没有那格拳探品的经和没有那格拳探品的经和没有那格拳探品的那格拳探品的经和没有那格拳探品的
# TOTAL TIME ELAPSED: 20.73s
PROMPT = "My trip was"
ANSWER_START = "\nGPT-2 生成的文本如下："
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset("gpt2_base_en", sequence_length=128)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en", preprocessor=preprocessor)
start = time.time()
output = gpt2_lm.generate(PROMPT, max_length=200)
print(f"{ANSWER_START}\n{output}\nTOTAL TIME ELAPSED: {time.time() - start:.2f}s")
# # ————————————fine-tuning————————————————————————————————
reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)
for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break
train_ds = reddit_ds.map(lambda d, _: d).batch(32).cache().prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.take(500)
num_epochs = 1
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(5e-5, decay_steps=train_ds.cardinality() * num_epochs, end_learning_rate=0.)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss, weighted_metrics=["accuracy"])
gpt2_lm.fit(train_ds, epochs=num_epochs)
start = time.time()
output = gpt2_lm.generate(PROMPT, max_length=200)
print(f"{ANSWER_START}\n{output}\nTOTAL TIME ELAPSED: {time.time() - start:.2f}s")
# # ————————————GreedySampler————————————————————————————————
greedy_sampler = keras_nlp.samplers.GreedySampler()
gpt2_lm.compile(sampler=greedy_sampler)
output = gpt2_lm.generate(PROMPT, max_length=200)
print(f"{ANSWER_START}\n{output}\nTOTAL TIME ELAPSED: {time.time() - start:.2f}s")
# ————————————tang————————————————————————————————
poem_collection = []
for file in os.listdir("chinese-poetry/全唐诗"):
    if ".json" not in file or "poet" not in file:
        continue
    full_filename = "%s/%s" % ("chinese-poetry/全唐诗", file)
    with open(full_filename, "r", encoding="utf-8") as f:
        content = json.load(f)
        poem_collection.extend(content)
paragraphs = ["".join(data["paragraphs"]) for  data in poem_collection]
print(paragraphs[0])

train_ds = tf.data.Dataset.from_tensor_slices(paragraphs).batch(64).cache().prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.take(10000)
num_epochs = 1
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(5e-4, decay_steps=train_ds.cardinality() * num_epochs, end_learning_rate=0.)
los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss, weighted_metrics=["accuracy"])
gpt2_lm.fit(train_ds, epochs=num_epochs)
start = time.time()
output = gpt2_lm.generate("春眠不觉晓",max_length=200)
print(f"{ANSWER_START}\n{output}\nTOTAL TIME ELAPSED: {time.time() - start:.2f}s")

# 22.6G-1.1G




# b"me and a friend decided to go to the beach last sunday. we loaded up and headed out. we were about half way there when i decided that i was not leaving till i had seafood. \n\nnow i'm not talking about red lobster. no friends i'm talking about a low country boil. i found the restaurant and got directions. i don't know if any of you have heard about the crab shack on tybee island but let me tell you it's worth it. \n\nwe arrived and was seated quickly. we decided to get a seafood sampler for two and split it. the waitress bought it out on separate platters for us. the amount of food was staggering. two types of crab, shrimp, mussels, crawfish, andouille sausage, red potatoes, and corn on the cob. i managed to finish it and some of my friends crawfish and mussels. it was a day to be a fat ass. we finished paid for our food and headed to the beach. \n\nfunny thing about seafood. it runs through me faster than a kenyan \n\nwe arrived and walked around a bit. it was about 45min since we arrived at the beach when i felt a rumble from the depths of my stomach. i ignored it i didn't want my stomach to ruin our fun. i pushed down the feeling and continued. about 15min later the feeling was back and stronger than before. again i ignored it and continued. 5min later it felt like a nuclear reactor had just exploded in my stomach. i started running. i yelled to my friend to hurry the fuck up. \n\nrunning in sand is extremely hard if you did not know this. we got in his car and i yelled at him to floor it. my stomach was screaming and if he didn't hurry i was gonna have this baby in his car and it wasn't gonna be pretty. after a few red lights and me screaming like a woman in labor we made it to the store. \n\ni practically tore his car door open and ran inside. i ran to the bathroom opened the door and barely got my pants down before the dam burst and a flood of shit poured from my ass. \n\ni finished up when i felt something wet on my ass. i rubbed it thinking it was back splash. no, mass was covered in the after math of me abusing the toilet. i grabbed all the paper towels i could and gave my self a whores bath right there. \n\ni sprayed the bathroom down with the air freshener and left. an elderly lady walked in quickly and closed the door. i was just about to walk away when i heard gag. instead of walking i ran. i got to the car and told him to get the hell out of there."
# b'liking seafood'

# GPT-2 生成的文本如下：
# I love china！
# i'm a pretty nice person, and i like chinese food, so i'm pretty good at chinese food. so today i decided to try chinese chicken, which has a lot of spicy stuff, and it's really good.
# i was in a restaurant, and i was sitting in one of the tables. i was in a hurry, i thought i was going to grab a bite, and i grabbed a bowl. i didn't know what was in that bowl.
# so i grab the bowl and i grab my bowl. it's a bowl of chicken, and i grab it
# TOTAL TIME ELAPSED: 22.81s



# GPT-2 生成的文本如下：
# I love china！ and i love the fact that it's a pretty cool place to live.
# so i was in my room at work and i was sitting on the couch with my laptop in my hand. i was sitting on the couch with my laptop in my hand and i was thinking about how i could use my laptop to do some work.
# i was thinking about how i could use my laptop to do some work.
# i was thinking about how i could use my laptop to do some work.
# i was thinking about how i could use my laptop to do some work.
# i was thinking about how i could use my laptop to do some work.
# i was thinking about how i could use my laptop to do some work.
# i was thinking about how i could use my laptop to do some work
# TOTAL TIME ELAPSED: 58.81s


# 欲出未出光辣達，千山萬山如火發。須臾走向天上來，逐却殘星趕却月。
# 4873/4873 [==============================] - 1136s 228ms/step - loss: 1.9375 - accuracy: 0.3894
# GPT-2 生成的文本如下：
# 春眠不觉晓時暮，曾解風光滿處沙。白日黃沙深處渡，白霞濃淨暮晴沙。
# TOTAL TIME ELAPSED: 13.01s