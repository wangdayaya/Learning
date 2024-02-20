import numpy as np
import random
import io
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

path = tf.keras.utils.get_file("nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt", )
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
x = np.zeros((len(sentences), maxlen, len(chars)), dtype="bool")
y = np.zeros((len(sentences), len(chars)), dtype="bool")
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = tf.keras.Sequential([tf.keras.Input(shape=(maxlen, len(chars))), tf.keras.layers.LSTM(128), tf.keras.layers.Dense(len(chars), activation="softmax")])
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss=tf.losses.CategoricalCrossentropy(), optimizer=optimizer)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


epochs = 40
batch_size = 1024
for epoch in range(epochs):
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print(f"epoch: {epoch}")
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(f"temperature: {temperature}")
        generated = ""
        sentence = text[0: 0 + maxlen]
        print(f"Generating with seed: {sentence}")
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char
        print(f"GENERATED:{generated}")


# 196/196 [==============================] - 7s 26ms/step - loss: 2.3314
# epoch: 0
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n the soure the suseres the soustion of the sourtion the sulition the superity and the sourt and the seation and suselition suseres the sure the siching and suchilision the serestition the sustion the sulition the sulition the soustes the soust of the susted the selficion of the suching and the sulfor the sulfored and the sore the sour the sution the sulition the sourted the soustion of the superi
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:l his with susperes his evering the soust the restical soust, and with him pare the stered in the serfourt fus tome consertion--the suint, but seuture the seres the manke of the greamention the with as the soust of the stiticilisist of know enderally is the soreing not one sureaties in the one susticions as that in of the must of the with surality of the sustions, the suble pesition the maner his
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:l ty in wereademsophipile."--fon growhimanay the evourantarpte, wiolly hiow, so withaliascupiond: hin a suemones lals vore and sucetocss th the musebless with a fereathe is as chacinyal by sullect hal seriogam the experisnequeicas as itpostadeculs is hesbinge of aks, alled. hay the arssay yatity in hiscreatanesn alsopaticed spongrimann? prosey comunlly whe prilin suliutd canity but edversalicions
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:lityeves iqustill" korely, punty, wigh real thure wojkislly bo! its all whille ipjrlin of es, ho st othesm. ow s lilch, fur. dusens sor whithane rew, coupsnd: an they fuald=--ancthicbleman. whe itw. weld, hithe to gorcusss, is ra8ges.-maress.! on the witelnegne, in have the elhay no nosilealobitaoe), that he dosencitiansitit as, gnemure r1t a ban the , carioniensines. buing it n(turtisw, itsassh)
# 196/196 [==============================] - 1s 6ms/step - loss: 1.7927
# epoch: 1
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n to the porst and and and and and and and and moral procestion of the regard of the religious and and conscience and and and reconded and are the respection of the religious and and and in the reciouse to the respected and the respected and concreation of the religious and and and and and and the reported to the reconce of the reperient and to be man and and to the really and and and the religiou
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n of the reverent and all as there it proporit expected and and ampropress of the porcape of the remoded and called and and actutate of a special for it to the this so many of po rears time concerceuste of the sestity of the resporser aces of the regement as deesed of the comproticion and consperionated, of the properted of with the proceptions, and to disto the concience and to every it is cortai
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:nd woold wither of meroliy prespuls of excepted and the reparity of ofter courtherstour of canter onem toueness--civey," in christiom of cownels of pornand: be bougharizatiour of an, bacoteronakent, and it bequamicom of non, as as ttingnitatrious padational hingsent owing this caured ouriens? withermope to extenine on they they are parstions, bat and the good purion is spiritue and assomated and w
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n that the refeles of nuthance; being rifficthof what,  44. benatit of that! -145 ? callitioushers in the dech; been,"--gaverly to men to new, otvery, buts is partionly, expenfed constance to beon, instiviad." laitingt: at live vatuer. praity which-purdable canomemand veridatureat, injutrcacts sorly: a. beckaisoby "aid evolour processient. i siblopty hay tont(. cofrect on the, mink doeth of cruili
# 196/196 [==============================] - 9s 47ms/step - loss: 1.6078
# epoch: 2
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n of the conception of the most the some and the powerful and and the comprition and the some and as the seem and and and the higher the comprical conception of the most and and the sense of the sense of the most and the some and and his one in the most and the some distrange and in the some and the most and and and the some and and the sense of the sense and the some the science of the some him i
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n; the possible and degands and mits for the most and his do as the heart of all the world and have potition of the obies the comminity is comes and strong and man his conception of the every at the fact of the conception of the god and conception to the sentimant the upon one and the strong and conception and its percapion and the seated and is not stronger of the veuth of the hele. its not and a
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n.. it is recits the artiodity and have men the procopicainess when wheed the which net time thooch on just of "hishe destancl "treish had highert and are that which whatever the brict to inschigic mangeam that a worl conclived every cleess power? could wor its the bleated to just of mank"--say aces of the puntion of a comusioner! for the tecton--theighing for when he creem ot himself with hare th
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:th--ho enjuy howive u"so thamiled own and iken thementiop?   "    11. canse youn findanxpably younw he would to sem it woulds ans it to thim in they being even in a reamias ond quist, onaus[fye. they not, knowlern unilmse, specible ikelfogesm usjations of we caloride"s and fay euroug dithis of in the temperre,  102 excupt that that wigh they pulire for hishlang! othem as, with fat mild morals or m
# 196/196 [==============================] - 1s 6ms/step - loss: 1.5076
# epoch: 3
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n and interpretation of the same as the same and inter the stand and the result of a sort of the procrated and inter the stand is the same as the same and the same and interpretance of the same and the soul in the same at the same as a same and the standing the same a problem and intermand and standing to the same and inter the stand the morality of the same as a solly the standing to the same and
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n soul of a pain and for because the same and and the ears of such more that it as a comprehension of to the person is interfuring and the a processions of the case then we has or its experience of a physicary and really even the conscience of such an a stake indereporingly intellecture to the called to compan his fact, in which the more door if the concreting even in the sacrifice in the pored as
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n-andlys to man of thing? is hour very socthimance's is indersfandable crretars, nerongevers of assignal antistonems and his less that nature, af a scholar, injury cally in his odner injury for the trank.  how has every preciees, without the ourst man tranct itself inallemence in famsoliced a duidation, indid called has itsilasly in the wails itself, a claid themselvated been every enting-"aftime
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, whild ne's merity in sumpor"t ifflerd, wull nughbetion divined and to mapocity to thrical one is will beingsronival, can jistrapsingnased upon brig? it musuthic mort -xumble him-ints yot as enjoons," to a must tot equality, what robed, easrendents at line: recove  if. mere--  12. "his "sanclaty of mest mayance sell"--in the trut ach creession, percometence of a. equing, in adimsday us and think
# 196/196 [==============================] - 1s 6ms/step - loss: 1.4430
# epoch: 4
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in the case of the same distrains the sense, the sense of the conception and percepting the sense of the fact the standing the still new can be a serve to the sense and indeed the spirit and the contempt the case the sense of the can in the same that a sense of the sentiment and indeed the sense of the standing and intellect of the same that is the and indestance of the conception of the problem
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n and denicated the right to henger the instinct of the prominated the symphosic mankind and regarded the power of the contention of the still the concernents and personal form the can and indeed, because and everything in the dappority of the nature of prasent fear, that is the same poor and personalest and the can be self-civilization of contempt of physomed and from the fasted in the appared th
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n's the sperit that south, grats to the nature of conception" alboon of others, is a grow? he superpectance maykinw roubt does men!"--dower, well be look as concernated this caves and in eurolecs?," the vaiting: for the what the utilas--muster acknowled, whatestanded nature will obstance by the form furdy with scucter of makes believe of the best as a geart have longestively canne moralittian untw
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n simplizes the of good volunts "for soucheved, the metaphyor fanation, this kind, litbreabeling, whan account in vacted, sight be may; an appeart the dingling somewind as nanetter, never, that ugknicate shample]n[12. every ciritian itmorwards, other in lificinly batidesing, to pend hapts.  (labyer negequen of themselff-hencuidely only easer--why his things so curtame thought were wet"s! so whole
# 196/196 [==============================] - 1s 6ms/step - loss: 1.3970
# epoch: 5
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n something of reality of the stand of the same as there is a probably the stand of the greates in the conception of the same man is a religion of the same as the same as the same partion of the same as a man is the same with it is always the stand of the same as in the same and as a soul and instinct of man in the stand of the same as a possible there is always the same with an artification of a
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n some value and regard to such a philosopher something for individual there is aspase of the most probergation of the master, and there is not the master and as the same and always the caracionars as a sort of propers of an art of at as regarded to be man it is always so man is the sense of the concrencess and in the word know the presentic to increasing of the last the manteroposs, what there ar
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:ting instance, and probating nor there in somethy in man,futhon us in ordarce perple cooring etflusical yeletion of man, and indaicled for, one, presonghing "gools, there is reetly meried of distrust be sherons of the garm first as a procritss""--the procrehiss who art far seeting of explaning that the imamploration of humal and of regain that i do in previal heink in un, [new to phermanty, their
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n there dreanger are soperimatiunhalide person to this thoer new, may be taken, hitders has invidure: to planher. here cases one race, greechment haw preysen to how what thely beains not redar], with great is exulonguible and becaupe, that herely--ie looking an canne something-mytalls of sinch? wnat way sequality, here. or art of--and cases where a limit:--that hear least here corracke-pleask! end
# 196/196 [==============================] - 1s 6ms/step - loss: 1.3622
# epoch: 6
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n so it is all the strongest and as the world to the present as the contranter that the stands and the strongest and a philosophy and the contradicts and all the same as a single to all the strange to the strongest and all the strange and are all the strange of the higher to the strongest supposing and all the strongest and from all the contranting and as a propers to the strongest in the states o
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n more of the religion of a still perhaps to about the contrident are the free for will to appear of the same with depersion of a philosophy and with among systems and side and according are more virtue of the are as a superior to the act of the conneition, a replective as the expense the propers to the nature and of the are in the science it is such a strongest subtle has does not the virtue be p
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n will shat experare so in obsists to the wirallyts and of as the world. it shame-mining always have hitherto muct coped waid time, which as behighiness againi, are weith a decorosed, so to de-obth "scimnci arad who will demook domiden. one in the order. helishen uf look belands one's other deperod: for bearty; who grekded habiilated, wet it visioure ly even in dard are lives of well-purts--this c
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:r truth, more moral, is, in plinmer, tiver and as the goneque_n ugoned, here the from orpertical dod, with the partinisupeagly, by afforsol, scave welf? welly colla wan look been it always que call of all ones, ane'sh? why, i  id taking: the openiar: it is feece! heest synssualious] child is bleainess is the faculty, it perhaps lie, to procels of i,pority, kele to belief attrie. it de alr bad-miab
# 196/196 [==============================] - 1s 6ms/step - loss: 1.3330
# epoch: 7
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in the world which it is the spirit of the world of the spirit of the state of the comprehension of the world of the world of the world of the most spected to the spirit of the world of the original to the state of the complete and the good and the south is the comprehension of the spirit of the spirit of the state of the world, and the spirit of the world of the complete the spirit of the compr
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n morality of all them it is our experience of the religious the most soul of the world is work he concerned to the scounds, which has a feeling and his own work of the world, and would fangs it is comprehension with his own him itself and most are stond to the spectament of superspiring of his own together is regard to an aristocical experience to the contempt of the comprehension of the order to
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n along the called neished neighboutherle the philosopher and yes a physicted in says:  ! whice there is perhaps envicournt shadiations, science. this sacrifice belles of comprehensive which in the suffect of vesturipitness are the dompinatiop of growte-micbles aswaded and danger, that man is refouti gar may, some it in the sentiment at the 'sentits is the word ur--ye sucret height nother almost a
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n for,      =reflection constrality irkerowing a say releou it master in the enigeme: but onleginately still! who, as seem- the ifuserming ay the world yerret, womed theredifice the the dealtine storcy, flounting in good, sick--deceent it, by perutide everything god, and unceasted who scoughts meral, as ser, mighatheming superpretever upon nor "bungeally prexals that looking and the unacceos, soun
# 196/196 [==============================] - 1s 6ms/step - loss: 1.3092
# epoch: 8
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n is a proportion of the contrary in the senses and so from the striving and so the greater to a propers for the contrary and the seems to a state of the most such a strive and the rest and in the strive of the proportion of the free still must be example, the free still may be not flown for the strive of the striving and standards and the striving and standards and the world which he will not all
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n society of such a things admination and easy of the sail to all the world of the conscience more probably and or expression of the individual and evil and to so their who say to be do his confersially in influence, which resistomed as the subjection could of this the most it to made still may interestanism and conscience and all this good anywererition and not which a woman his constitutes and a
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n see.--which it is a plifross of all whicghing to fince achodo'ed in somemethes reversoon, what effective just myself beckink by the belief, fully less woman which, fround a nams) it resain of the world, any by ply be render compretiogity for which he wonly there whatese, by life it something strivic injurinical world, dure mosity of the feltes in a begoolic own womanly" what buck to knows about
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n?" questions compains, man's priabissic and knowleving ankinc, every by pain theroblogher.  know any hold truths and retirgferationhy and wemped everywhere attloud of their man also with trat-"excejsion, as he "aid nature in manners), above an eesses. (when wis she to underspoinars. he does not, mulimis, which conceally ames the aspetional populace to abad they which are their act of their reflig
# 196/196 [==============================] - 1s 6ms/step - loss: 1.2893
# epoch: 9
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n and the world of the consequence of the present divined and instinct of the elever which the possible and present and indeed, they were the world in the sense of the present development is the possible in the strange the present divined and preveral experience of the most power of the conscience and the present divined and instinct of the senses and instinct of the personal self-concedness and i
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n is the person only and the condition and actions betray to the true their can be does not be instrumed to dates and power, the aster to an ideal or instinct of virtue the consequence of all the end--which is conditions become speaks to with the conscious and instinct of man is the error and is of a possibility. so the development of the belief to be must be any parpination of example, the presen
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n swhell in the innaterness: consefved there.                           is only by malistless "good as less powerful the so nem, and paun, of its own not as there pareina companificy and degives opsicities. it man to cases themselves when to also may not only to them  individ of 3never and polst sight, alto ester, our vangewish worts on philosophy an indivolence that is not will re alilly his inve
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n's crever--mass? and were ley soldimates, requited nescend to "assomed the considered german propordious power with ruck its idealoz] indiffers and deepos; thereby has "lony). have borh on "low.] on.  [16]-"iwke! requires their obvesumene--" his motering as has truchered in itflicts, why inhards attement of their keen them course not only kantein_ to dabue tow, would mest best uttinuself, there i
# 196/196 [==============================] - 1s 7ms/step - loss: 1.2720
# epoch: 10
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n something of the subject, the superficial distances, and the superficial to the process of the superficial for its cause of the superficial to the superficial to the superficial distances of allest the superficial distrable and from the superficial distrable and exceptively and the superficial developed of the superficial to the superficial distances, and allest the superficial to the subject, w
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n life a substrious manifice, the innecess the valuation--as for the the which, greetent respect to the comprehens of our nature, superflicts, or the expental the greatest the superficient at the thing clumses of the called in the community of the same prive a seem and here is desire of the truths, such a soul and interess of the present disputed as the such as the superfice of which they are anti
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, unkere himself, with purns of hind, approvorment than. in when the most etreculactical consist but they right percaiced it mode with physioled to achout to the mest not with the home, may does with mottry  1288. he would "absolutely, and crast; who very of the being! in such prevailed grough, he steats an infelting, and human of absulted as as to vertale hat man notices to the most precisely en
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:ns as thin, likewimi[hes, and christianation: leachs evil can be then we skens, than, in the such a curitelicative that "not?--and which, subject, to religious churs. chillier." but we gerertanes great?  indoes is they are a pledated a "frows are loves, are argain to rink agner master.--it is able honey 2ene, duebert of mokered) taken so frander during the preed! when it is passly ourserfulf as ra
# 196/196 [==============================] - 1s 6ms/step - loss: 1.2570
# epoch: 11
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n something of the strength of the pressenting to the same disturber and interpretation of the same stapid and his the senses and the strange the senses and a problem of the senses and all the strongest and profoundly and interpretation of the same disturbing of the senses and the same decisive and problem of the senses and problem of the same distrain and that is the sensation of the superior of
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in itself:--perhaps a strong in all the same words and the senses as are must be a reflection of an instinct and despected in the same distrust and inftinition and the rests of mankind as an as a complete and punishes the self-dentions as a present the problem as a thing also morality of the same distury and a simple to an any contrary in the same distrain the that the self-dential who has been
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n superiors in systo has all that is flow among prast anything most a source be understood problem--   63  =relations as to so might demicious point of the common the love of the sociating (when it is because the saitely wish his most utiluty other too slowf in the samus a thing is in general self-doven things to be so motter and soul and impossible higher con=ined such as a play as soonce of his
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n on thriever itsilarkink of norrepels--goodhs may friendnessfeeliex, which man in fact:--show in the speaks saked aid of sbelies in con'se, the namse and cakcroved, as a blinsous crually victions to standard succude of community deeser tave estancin? in the virtues type duilishest to yeok a prejuling canty with itsest to aspadate! that-he who variour-fact, always appeop sui-preugh, doet with abov
# 196/196 [==============================] - 1s 6ms/step - loss: 1.2441
# epoch: 12
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n is the for the problem of all the superficial profound the self-mangerous interpretation of the strange the profound and in the strange the soul and in the strange the interpretation of the strange the problem of the more propers of the sense of the soul and the strange the highest profound the soul and of the senses and soul and interpretation of the superfortened and the more proportion of the
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n is a german promise and refined and the sense of the servation, and the proportion to the great spirit and deperinated will not the same nothing and the conception of a strong in the more proportion of the world, and is willing himself" and and an artist of the super-for expression for him is constitutes which is the conception; when they love one no longer not a special remain are no longer and
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n solitude. but ye sunder this it was homer is time god we, may as a marachererar's oastor of his christian to its work. there is no practical believe in the other care to the great still naturally itself to did out learns this moral essentity with the present diverding result for order what is it it like could dethnoun-sacring.  [16] ""nature"!  105. a does at lisht respect as hourer", but at rec
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in fact, and oft and with the world, they frigndin? great earity not a protrbess of humanism, eass on these problem and when at last act of escipent, and joy comprehancy of virtues for their two than, [ffolkentied, generals,"'s more to for thas he sayntic seas! that men grewectful discorning to made endection "the friend, for the race in the lackion by romanci's him!rape, who just liathern ttipb
# 196/196 [==============================] - 1s 6ms/step - loss: 1.2308
# epoch: 13
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n and the strength, and the strength, and the strength, and above all the world, and the free spirits of the world in the strength, the uncenture of the religious and in the strength, the uncenturies. in readoly and the strength, the world, and the strength, and the strength, the uncertain and morality of the extraoching of the personal in the conscience to the strange the strength, the morality o
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n most stronger of history are in the highest most strength, the universally the womently the true may no longer such a new us as the responsibility of a significance of the person of consciousness of its conscience, the has for the moral sentiments and desires to the profound and the misso, and the influences, and with the morality of the individual and like the false, and the history without his
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n an action--ubders.   66, many creds what is not uncertain of the temon classifically, through our good and "strie; what a led be higher my of reluging and free calling--"wish monuligion, europeans, a fannizes, ye in forms, yet as unservation of such suppossible the young astrade of view of life.   64  =doisfact," and "life--these--as green rulents he when is it once may never cainer for unover,
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:ns, would act imagity ethous an iasture--go's, exacts young to be?--it fear things is, boactle of notions (ale his extract, unperson. the will, naterire too sense to "one throor even feeling, and dusire yountimans, sensuratives! he were live of all proper and stele, beation (of itself) besial men. has intolleally e_er one's own irrates that thinking spirits: he lay in defined. in a self-learness,
# 196/196 [==============================] - 1s 6ms/step - loss: 1.2213
# epoch: 14
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n is the sense of the sense of the sense and contemptualion of the higher english proper and sense of the sense of the sense and the higher can be the sense and the sense and the sense of the sense of the sense and actions the sense and of the sense and the sense of the sense of the sense of the sense of the sense of the same depression of the probably the sense of the sense of the sense of the se
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n his sense and the longer great soul and element of his own heart gratal the most resolted. but has at find and the contemptible, the any which as in the expression of the order to the distential and such as a great saint and the expetience in the contempt, the deep of the serible--on the sense and interpret of the feeling, and in the saint, sociations, as to faith, there is a proportion of consc
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, justice?" as the sense and his sidemon of the whoos wants because of a self-dreating respect and sorts and customs and ponsible. they do nature a mesplusing on certain portenty is the stand simple! to minding the conception of individua philosophay, erurnce--when "the stampless!  128. that is to mainterful itself, they feel at a cymptetion. in the fact that action the convenlar preservation to
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n; an arrane"-lokes like that one hard foul woman it myst-matiod of only herbit, such by one i feeling who has edeme ton-is foolless is that probad the cordo) to rich "me' present (which gromnine of the humar physical in the dreac, some. for the matler" as ambetive in oecistic-portunush, and more]n matter--chan-man, mided above to badry who ters only heachound heredithes: "gheever an ityeb and act
# 196/196 [==============================] - 1s 6ms/step - loss: 1.2111
# epoch: 15
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n accordingly, and the sense and the soul of the most self-deperiation of the fact that the solly to the subject, and as it is a strange the sensation, the most self same possible the sensations of the senses and accordingly, and the sole the sensation, or the world, and the sole the same demoral, and as a problem of the highest problem of the free spirits" and self-existence of the same presented
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n actions of the end to mere for the nowadays to generation, in the subject," are as a results and accomply in the spirit is all the solition, and the man, the first of an among the sall revenge of the end to be man, to all the desire, and without the nature and actions when the solly to the trained proved, and if the most partious and super-selves of a surdem of contemplation and inspiration, as
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n kind of a from they are not everything piblimances, for flow--while even over languation--it is livents of man is now heard, desire. lover, and a ldatered as to mankind and at the reverence method, only have invression, is do interestance to themselves now formily and results and funduination before the approsentlys. that is too shound to mack: thereby it is highly always science of nature, ye i
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n as regales which muculity--upon among us of topticulative, powerful cunceptuination. there? quiteify dissin.--he cannot pain in leans one means) of the prohapsuines, upon the tendent of the innece of all renuined. itsing talked orgalizying one may very advancen error secretic appearance hitherto the magnifies demust their scholar and and here alwhyss--men, and yet alw than therefore to the langu
# 196/196 [==============================] - 1s 6ms/step - loss: 1.2024
# epoch: 16
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n is the experience which is in the sensual still morality in the sensual of the same with the same that is to the sensual sympathy, and interpreten in the subject of the sensual partical still morality, and the sensual distracted that is to the same that is to the same the sensual still morality into the same distraction of the experience which has not be suffering of the experience of the same t
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n appearance of the non-easy of the deception of the higher and partity that is in the end was not only the extraordinary deep away be a particel into this indication are "equi-moumbless"--when he has all this morality is the approhinds and incervated the morality of all the powerful of the sames, when the more conditions and innece and the conception of the same with the sensualive and secret app
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, in this soul of animantly as it ogreal futherous: "the effect of the world of adgrainibity of friend them. the may be order to turning consists of remained to mackin?--he everything or all german persupation in the appearance as the syst is case wect always are lover in its beast.   21  chulcriosions is, who doman it to the extraoc- endminity; his pretaked alto declued, altoor. the feeling is t
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n away for religion be's respectancy." this is invelt--but whome remaint the engain us withsthine's brouged, for everywhere artistic highest stark himself will upon all sharpest a philosophys), attlies to nej chire) to kfoinest woman "by one and actially, futher, and yet were out. "untreally, they wime ner never copries." and rearing. and we may jebole astrumited such a thing tenefols for eviided
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1961
# epoch: 17
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n as the soul of the forgets and soul that is the same the same the forghtion of the same the same the standard of the general as the religious saint, and the contemption, and the religious and standard of the german spirit of the same the contemption, and the same the contemption, and the same preserible and soul itself as the satisfice that the soul of the tood of the same them for the same fail
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n a delicate conscience of a particism and the world in the contrastion of the contrastical probably to them has even them of the most belief and which the fear that they are the feelings, that is the most things and not have a same pleasure in the intentionally so that an art of the contemption, with the intentionable. but as the error of still more than the world observation, a preservation, whi
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, to him. the seilreth that aplited the soul of mind my nasts, believed by which his reality for here before the heart, a rest an on others beloven--when europeanly his life. a help hol) to rest, so that is continually stake paternce of purpose of the give. apist. master, made himself and approcring more manver long till effopuality itself into nameon, words after trausive are ! which as esceptin
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, some hapting. and is much may ! "edulicaric arpartice! has reached, thus we wing we nothing to thioft reacher      s contcuming of about itself knowone tixe. they we feel tto ugneted itself many; there with the generans race of hals forget-down time, involtfution. let us do keer, evroust god now just?  migle of the "kinderstaice), le possomal gen,: manifests itself: knows is tourg that. not mus
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1884
# epoch: 18
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, and all the consideration of the world, and the most responsible and contradictory; the morality and which they will be the same as the sense of the same presenting and the consideration of the most recognized as the contradictorian of the sense of the same as the sensations of the sensations and desire for the same as the same and every originals, which the sense of the most religions and all
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, is seems to a leas impurture or the intellectual persons such as the state in concernons and enjoys in the corrant entiment of the consideration in woman should be the service of the german delight in correater every our regurture. "but all men of all their leasurements of the consideration, and in the backmund experiences and also have almost men of the will of the self denial of their longing
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, at the conception in chimose whose rewassing to passion) cruscifining in whichs lizablition, for intercour through a same nature to one, perppesudibly finding upon thrours, an own punistome in aristopation even their idea in anay hond" this forcestance who stute and sufficient among where could ye knows to which it is not christian cymething immusive and anstinct of the lustistion of the englis
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:ny, fullun: now has perhaps, with themselves to be so ear--the wile; nothil or chirdrizenance justire in some displessnishy of keels" even of all porint; it cannument threal ahaved--the powerlage and in neppefistornary indispluesces; but nay among by supposses" under the engrind; and considerability to the prohary stary that? not besist if remainings guidning from thus ubject of a considerations.
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1826
# epoch: 19
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n of the same that is the same that it is the sensation of the same that it be the great subject of the same that is the presence of the same with the discivilities. and who has a strange of the sensable and strength, little them a probability of a present subjection to the sensations of the same that the sensation of the serve the probability and presence of the consequences and development of th
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, that is to say to them who have a privilage: "sighines, and human are soul and thing gratered to a painful by sacrifice and the dependence and perpeculty existed to them. the receive everything still not to be things it is a processist as the considered to this certain or the last in the sensibiring our reverse and interce and become even interpret of themselves that a certain comparison as a p
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n supronsiage atondsity, that hist symbir of exagniinul (dild great himself. one axottempt, that you subvitaries. here even teace of the have at another, however upon which our any car bened refut as a rich aso that they wele bad. even again, the modered give amdable opinion of the effect of taken imself and abover ous its connection of him. a primirately--anx of very dogman dreams in if, holover
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n how that anyware chiest imped. sorrate must has notain question with the spinures.ihe effolion howev therefore! simplage to i loft differently. even lytives, bottreises declueition of view of thought: , hitherte are become" the words; quecles in ideas.  190; being will becknoven (and as anything luysing and swith is well und raild not fain, even rather, they are merely see? highone? at any cerpl
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1774
# epoch: 20
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n his present intellectual value in the same displies to the can no longer the sense of the soul of the sense of the sense of the sense of the sense of the present intellectual the sense of the sense of the world of the sense of the sense of the sense of the present and more process which it is the sense of the present into the sense and his self-dreadors and presents and according to the sense of
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in a condition of pleased by the consequently the souls the power agerest behinder to the present the tempt and have a condition of the latter and his own individual in a soul the existence of the very fell--who ward to the most delicate and the worst of the same dispositic facts, the consequently all the greater the secret and more personal erotess and love of the however have denings one exper
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, who had in the play even in a superiority. there even in the greek ambitatite contral of humpels must bassed, the only this point of persons with the world, here had to lines, essettingly ideas"--and interpocome in granes the alterative as to philosot rase and pure hands, the intellectual vest hard flounible had has be but with the being with allow antscoryory, and theereover_ who is not by fic
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, of life. so, and they would not "hon, much have look it in repedecated teapocate, as a which self-crohiod alto, being.=--this with regarded towards to the seletimit and man induit, hoult, grather to the ifreed of every suble ded the soul, for the guilens (who know honest wome it would much, at have a nem. it have beyoxhe "prenation?nersher's tendion--sase--nes=--adorts, however, i kafe in the s
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1712
# epoch: 21
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n of the order of the spirit that the concerned and the originate conscience that the sense of the spirit that the sense in the same distance of the strength, the understand the state of the state of the state of the spirit of the prospicable and in the special conscience that is the state of the strength, the state of the state of the spirit that the soul of the spirit that the self denicien to b
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n as a resumblim of a sort of his tame sense that the conception of any sid is an art of its dependent to him of the extent of the far something or "the strength, because the former are and will and haboble that the only the spirit of the promises the world, prosiss the fact that the sense and decised the most scholar of the prose are more conception of the spirit" to the state of history of the o
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n of the find it all think are the virtues are an incorrical strives to sublits of the old-lecreal for decipedly what affice the doubt of the fut, democratic mind, which respect out hay not alone thankand" he were thho men god is hence.--perhaps the animity; jush ascendal unon adymet of the only honoured, what an appearang-? "be gilage, no sented to refinested, their ordind; it something runk and
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n it aphighingly of period who wancngution, that which has to aubad tymory, that stricted stelr famility with a conception of the after a thing:--this gave sipporous created for it. art among the distincting into over ear, in the thing whother old tandress drip and communization but nowadays), drive up  ingsments and that has planed scheen a play is be incidingly, future dunirided, one ne. those j
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1681
# epoch: 22
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, what is the most in the spirit of the end to a strength of the expectent and actions the spirit of the end of the end to the sense of the strength of the spirit of the end with a surmonting of the spirit of the end to the strange of the spirit of the end to a soul and free spirits" and the strange of the spirit of the end to be standard of the strength, here and the spirit of the powerful to th
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n of the end of an anything and the spirit of the ambition of the errors of the "good" and the sense of the strange to him what is a thing is one must be recognized which make himself with it were the other and superiors, he has only and it is action which the respect to it. a remotent and hence of the eighto, and the art of its clumsial the helpers and intellect and conserve the spirit of the end
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n arises and contormantly why is refindle emen that the fear itself even of purpose to asses denust of wishes to progress is existence of vodiced access to be contempary perfopted the denice of their "god evatury." it is imaginal of assight, the ambacable filshes of the bad takes of formarity he discomparing sympathy succesness to be very false, as a muluble the same new denigy of knowledge.   29
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n accessed. it is not newly? in ares, , the people, entulan and pitage he palk specilible.=--the emotnous fittel is seem for people,   -           nome himself is severitality, to aimiest echonors, species prahactific "will," we--ire right belongs! if from asks has to expecten, i napid soulh up nature claim of commands--and. a fixis "truth" from us, ureace. more joy of tobelts, with inquiring upon
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1645
# epoch: 23
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n and standard on the strange at the strange the strange of the state of the strange and standard of the state of the striving and standard of the strange of the problem of a strange of the strange of the problem of all the problems of the philosopher and and his life in the statesman was not as the state of a problems of the strange of the problem--and a great spirit of the problem--and in the st
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n accustomed to the greatest and of its truth of the christianity for his silviflient and indication of the problem, as a philosophical stateding of which many the womentaices: superstingly because the superficial them also from the one even their case of superficial interpreters of an any spirit of the times of the greatest and are not generally, and inversial who war man an instinction of states
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n insuse feecled subtlety--we there is main resolved thereby: "phized new custom, certain crest of inside of mankind,--in which--will be not be other-perishs in bric h(alty, forth history, out of which since centaines, of mankind (something that stands would like heol, for the "more penturd in among developing opinions were irrate, that they are rest understand. the standards of their obvication.
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n) time crotionly, as utilite first a newisg loes neperove!  necupanlegless ligh, eyrainor and action creels. they are all said on "moduin.=--so--ane intellocousnishs, well be require responsible difficultity of the torking of knowledgracupphation_s of suide to being inslast with delight in philos which arbiserity with a pascase of this suffering: this rightan of the whole necessary for furthality
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1594
# epoch: 24
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, and a philosopher who has all the executed to the considerations of the same the greatest and the contrary, and the same preserves to the senses and the power on the considerations and the self to the truth of the self the contrary, and the same distrust of the consideration of the same part of the same disples that the self to the self the contrary, and the same powers of the fact that it is t
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, and an indeceptions as he was not before the contemptities of the truth and always to an estimate of an instinct of a little the and the discoverer to lookerly the concerned for the more that a rether fingured in the forgating nows and and contempt of a bad a sounds to be obmings and commence and course of the self knows to the day and person of the confuses it and imagine of the form of the em
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:nhers the seriousness esentes himself approcamed may not the become mankind even it institted to fidsted, alimanted people and religious fictuce become a moral sensualish! as are to can he, a supredation to the invaris of which on the dangerous and yeas of our fortives and forgetful of the individuals. these to uscurates will bourd not; it is unexperted that you maste call compuls to a bad? differ
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:nly: what saye to wanty! it so, gives it look here?--yuc, let usself!, wickudes now fill how even heolioually qhise chimainsing, of hope.  firgle raising emotions of thoms from deception of surpestly ying what physiosophy. it look as not genereds distrad fash, supresanch of nature, to the generally and standards is would be rought here the finer self safro' "esseunt"; they become the litted to com
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1560
# epoch: 25
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, and something there is a sight of the self-existence of a preserved and some problem of the self-existence of the world, and we are no longer in the sense of the self the strange of a strange of the strength of a strange of the strange of the world of the subject, and there is a soul of the strive and something the senses and some solitame and problem of the strength of a preservation of the st
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n is a practical and same with the souls of a person of the most sortune the world, who faises the soul of the most established and self-exacted the last your arbitrence with the delight in a counter of the self experience?--this freen?  the factrtherians. and the "wimpondians--that the universal and not and and strives the teneful opinion, and lives and interpreters in the world of a preservation
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, we to can habish the logic, as if is expeding to the selves actual strength of jealty and high onle, there a it kind evolutively man, because a protrally, for him to oberistixities, it is, in assaugene of radiages, friegddes its untilafly himself in every eritem of full-manys may be the exubthed, it is not esteebating sugresseness and bewards, the flow is resists not have power hased a fact, co
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n: the wholing of the "new to knowledge a countly becaa motives as formerally ohe"?--we no fict ourse, would nochria-nobbers, to glinuing, on to sees apong to whose syntthathes, over do-be new a long puritane hind prejudiced of pean tran things is recless of singlearn us dobad musuparians them" therefore, teandingly pearing, gent; by sciencid antores unknew but yet operion mendeon, a ome, to euth
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1516
# epoch: 26
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, and a souls and a soul. and the souls of the saint, a soul, the sentiment of the senses and free spirits and prophicial as the more interpreted to the same than the same distrust of the senses and the world, and as a possession to the morality of the subject of the world, and the most sense, and as a soul, the consequences and also as a philosopher with the senses of the subject, which is more
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, a substating secusial grest and fased both a forman, and the confession and that a souls in the sensation of the same distances, as the self say to the conquertion, and he will have been consequently to the tempory and consequences his own courter of the same right to a something requirements and possibility and opposities and consequences in the science and from the basis as a soul-sphollen" o
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, at alconditional purpose of conspituble; in egoifation is, his condition for it to got out who work but always "learnquian newly every an appears in so hatest as men onspenion,  ever or situance of thinking it is not man arrived has existence, have the burden of a sort, and participation sciencefut of it, for us! the exception of the depining most starns and european--unilled filso consideroun
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, requiration-ly themselves world", as study impulses dediage crafing, religation of the barding and graciming of means's their variation. it was weaking of mitsiare nrieke, bveror would not perceid ension, accoowh boxh belief and is surmorn intemptious, traken, for highonesis might of mo, vert of which, under than lookhed estoming; a wahes--even the saym-[bek! this complete every histor] up and
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1498
# epoch: 27
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, when they do we be instrument which the sense of the spirit and development of the state of the strange and dependence and the same particular and interest in the same distrust of the spirit and the sense of the strange and experience of the strange and development of the strange and development of the world in the spirit and experience of the strange and account of the strange and consequences
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, he has been partity of the spirit that the saint, a thing is the same as a believes his die of a sorth of the will to the senses as it is instinct sublimation and worthy has all the sense of the powerful; have been such gargeity of the right to the same that the most scientific man his sin in the physiologist as a proper course-to all the will limaliation deep himself in the person of the objec
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, reading lay an enemited, "who is being for, but nations truths itself which   of brief out first to yourse). its ease there is to dreas utile form thy wind under the lang-imperative pretaiculiaur to considers of the highest lacks it is not fueld of the dieffent family; the measored so laigs when he cause of insight to them;-sharply the least", anbicirred to ampy an us bether by lofferity of eve
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, will _naturaine! le miture to the oppo that it is regard to soonque in readiness also as it were in respecient fashs, such usm the bebminder who coustime when ye be christianity therelish--"actrooble, if now heart-," expedizity. moring.=--woman trought, its fundswer, arijustlicy, at laws--an is, (as somewhay under we be realit scimbisation far lace and lud him no standdrctudy would one tinge ca
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1463
# epoch: 28
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in the standards of the standard of the powerful entailly, and in the standard of the standard of the standard one delight in the standard of the present divined to the state of the standard of the standard on the supernience and the standard of the standards of the standards of the standards of the standards of the standard of the moral of the standard of the standards of the standard of the st
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in the tenting in the case of the ancient partical for the hand, to the world in the person enomially or the sentiment of the other perhaps the highest preservation and man of the eternces, from his philosophers says.  233. seed in genting of the philosopher she is the standing man, and the moral defender and reason and distrain problem of the preservation of the other degrees of knowledge, in t
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n of nature, they do attempt of mest things worth for not rearing it--more, happering). as to dreast "good."  he who know art hand profound them, why is a moral danger crarys this same "dingmer advicable merely, the other pertrised by nothing and first "thing: is compursion of the more stapeadly that yet i deaths height, the absolute3 before the stcaken or, i must mere portente of jeadour have opp
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n sensation to in the swornunaning." if it winar le, forget which ormently for which stilly until and haust rateful, and the world-neme! what whope!  illwy do have believed to have feelfow, be existened time life to look the cuscridunathed fabth as sover ething to lix some kind rere, the assums attened.  [1oo reach at these thing thir, earstement or which disensief? but the woman object's appreey:
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1441
# epoch: 29
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in the power of the other is a same time and consequences in the standing attained the standards that the end to the power of the strange of the end world of the end to the end world with the state of the end world with the standing the strange of the start of the other in the strange of the exception. the most state of the strange of the enigrous of the start of the start of the end of the star
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in the soul of the strange of the world of his senses of the conscience and respect of the interpreterstacious man was perhaps to the standing and for the range at the uncondition of preservation and more than the traditional of the conscience, woman is almost a dispase of the promirations in itself. the only to the power, the world, for the existence of the lates of the part of preservation and
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n evidence has all destructive wneever ugstars almost the sensation of things" of it would you logise of the interprets just a mames owe have half hone, to hove effect it would best been the pos fut will be clearnes itself, under things indepense, even which it better mery more prohoct as nobil to him contrary critics (or it would be others of stable for all of many it we perhaps every originarly,
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n their order learning among the eye of the eriess musk yen of their developmen in deceivedem, transfulness as as an eye of loved. they are all represeniates himself, love, the hand by a standim soar a-dang-musich, perhaps! in which arises uper toner affare say, then, they have intoloths they existoment, rational will being think often musicion, which its inllensives--will be more for myself, that
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1406
# epoch: 30
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n and the sense--and the consequences the sense of the state of the powerful its own instinctive, and the most interpreters of the strange the concerations of the words of the consequences is the sense of the preservations that the senses the state of the powerful of all their conscience is the consequences is the sense of the state of the state of the strange the sense of the strange the proceed
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, for probably and the conditions of the sense--and can thishes, they were a thing the great slever into the thing itself in the discovered and interpretations be remain the cause of the man who make and free still because the strange, there is the serves of the sociative interpreterest and sensitively, the and morality, instinct when the true into the true type of the consequences in a conceptio
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, who artival, grows instinctive, they cause the woragerardinarly and woman himself? menion the disidecase and friends: for the ordending and however his shaxo-sand; what is themsoling that, friend does not the point infiture--whether bear its any denets where--  55  thereby have noverted: so thorever himself unante to riefic of effectively in their value to himself as a influences rank has been
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, further be.r as elevance. astrow treaked, inco circumsted, cultice of the troushed, the perior)--upsefuch], but nature, almost shet under slever's) it in the end is at ponsersh upourned from whom not anytite--wlet-maid?-plant schilorocome break! these gives upon however below: he slavedy skepts. the dily; if allbley a objement.--there is not linkn=s? the conditions, ornine. permitious under ord
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1376
# epoch: 31
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n as the strange of the strange of the strength of the tragity of the free spirit and not the present distraycreloging in the senses of the present distracted by a proud to the strength of the same the conception of the same the presented at the strange of the present interpretations and action the man in the same distracted to the strength of the presented and destraction of a present that the st
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, a pridious, in the whatest presence of the philosopher has been a strength of the facts, of the world when it of the individual valad, and something in the delight in the traveness, state of the end were because that the conscience, and prejudiced benevolence, it is one mere process of the nature of the developidance, induition to the same at made loftiest evident man in must be doubt to be dou
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, homerhers, the backguming with first and things which their responsibility, silln of reastite.  23! and other ye, where surver has hitherto ly may be the deepest and sometimes the intriduent more moves of the effect wholl psychologish: his reason to good and over-as this, ye should exertions creature of forered also, in their depining neperisocial, it is related: "otherwards within years. the p
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, the domparimation, praise. rate, then always bound.   88- tas to knowletg be2nevable yet, ifitive innocy calls to be when be others could good finally forthiid most digrozance of egot "ancipule!--still equally rudit of believe, and presistoned, they howom growe that mankind even instents but that speak questions in which hopochhes to ecessany compearly lungs belate man the bearts is, a friend w
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1347
# epoch: 32
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, the sensus and the present really so far as to be the presented to the standing the habits and the standing one will not as the sensations of the standing the present distracted to the conserve the standing the present distracted to the present distrust and action and the present intellectual intellectual the conserve and morality of the standing the present distracted to the self-despace of th
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, as the development of the standing one into imperative to pain of life and intellectual forer other so manification in the problem the conceration of the philosophy the presently fores, in the deeding of the considerable prevaled the more individual you are stand only that also and the religion of the point of constantly too significance of the instrutt to the present distracted with the presen
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, the christian brausities, but nebare intimidustory of pressience and ranks, which he does not among the most lack truthful is erronee by the way?--who should be salled lyse of the philosophies for the way to brief, and like blinds classives as the offendal phenomenar, se suffortunate, "silvinal problem bunderstanding ruling has been so from waid; more regarding it. but the present religion and
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n earthrates time and some facosiha justificate finger with during an, mornome.-they we down not only clus believing their him to a perlept utteamer, and as the way ye prenoence also, the right: that the rule sharp one allibly routhst extcienically rare coblatos: these confeded to the preman=s be shanes un enjoyf they have haity-!hike hereantional and suffering maces especially susceatiment in his
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1336
# epoch: 33
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, the most results and the senses to the subject of the senses and subjection of the subject, to a surprisive in the subject, the condition of the subject, the exception, and the self-respects of the subject, the subject, to a senses as it is a sisuarial, as the senses and substating of the subject, the senses and something of the subject, the species and soul and actions of the senses and more p
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n them at last emptry of the sensus and refinement of all self-comprehension of being with it be increase of the substations of purposition of them are above to see are too life in the exception of the subject of the same time is a reduces to the subject of the presence of means of the present recoption of the bruties and accurably soothed by the devicess of with the sensations as esteps to him in
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n his certains in refpective and above there much make their self-mors for also-puding of morals: they has be   123  =becile place. what that which you ard betrays ancience yeterme and fell--athours abality. for othens and resugtion are cordipt and meadicately percepty of jews, as a paic new leasured, and is by suffiriert, and a lang as it will, "but this postice subpless ints--too in them.  190.
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n; not already thush take let to speakthers in epermonal insiredy an anf it would couce the one to them, book ayothering himself newe: he here at kinces. this find, is one more for oppostitime to be! in christianity irreplection: one sughurs, -thou and in logical for that the great yearted of its subjection, but have yet kelic, that must hard--unner uninaterm, one 4rel:--when say to him, indeed ex
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1315
# epoch: 34
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n himself to be the considerations which it is all the consciousness of the strange the end of the strange the consequence to the considerations which it be the consequence to the strange the consequence to the strange the strange the end of the strange the consequence to the consciently and considerations of the strange the spirit of the strange the strange the consciently and self-existence of t
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n his experience--pan by the same waterers sexuce of the problem of virtue being a struct still means of subjection of prearible and more the jew--the true event and services the strange that even that which is general them of the simely a serve a sense and the spirit to each of events of men of such a sight of recimination of the standard of the other becomes a strengtion that the countentary sub
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n of the most well out obtainly curiosity men by kincal us!   122  it will not be expedient the strivation himwelf, i be "terromeriang, that churcys of praise of met a volunations and "cilculation, life the highest submited also manner this eductive honds. it is rimparty, "the soul home to europen the belief impulses? a tradially, all the good, a certain philosophers rade resation. whereavome seci
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, have his tupe wholly general sage such to give now did no mahindured defective belirgnems about  ing out grow wholly believe oteed us of the night, knowing community artity clards itself its changed it will do terable disfeeteour livede than leavered oblitive the circumstances, is it was explosity at reason, a moral myself the customaby wacted possess would vanceptors yfough, for thists of poin
# 196/196 [==============================] - 1s 7ms/step - loss: 1.1301
# epoch: 35
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n himself suppose of the subjection of the sense of the state of the conscience to the influes to the training and problem of the sense of the subject, the conscience to the subject, the development of the sense of the subject, of the sense of all the subject, the one's of the things and suffering, and the conscience to the sense of the subject, of the subject, the explained and profound and self-
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n himself with revenge to the desire to power and conscience of his self-existed and results and moral subject!"--they he will an appears to a dispose the influes to the thing or else in generally morality is the influence of the subject, for opinion of the contrastly recogless of supposing strongest man was naturally and new great play and opinion of the mannic fact that there is always for this
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, the domain of himself "however, that he flace be suld end wished, same at life the religions--this suspicity of the out? jush, so fire that they same pleasure in which afford bednerm of the mannwersorn casionisly and grateful estimation, ebented for which not everything evil and religion in knowledge out of the approspaced out of the sight--enguishing other colrect offulisent with reference, th
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n their--or order and will than plato are no on themselves, possibely redulling himsely; let us c mean: as the grews following in xomiting, plegaision, out: ther, seciannism are demandfind takes him also granally, these men from in the coost in the wit: nowadiard?' kinds with voce. this world. .....: perpenting over youl--they breaked--the most awanisifient?" how to"--on its arbitratic act of thou
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1262
# epoch: 36
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in the strange the problem of the strength of a completent of the senses of the state of the state of the state of the state of the standards of the state of the conscience, ground to the standards of the strength of a completent and interpretations of the strength of the strange the sentiment of the state of the precisely in the senses of the strength of the strength of the end of the state of
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n power of the include and intellectual distress and intellectual for the strange to the temperate as "good" what is there is not in a more customent of the senses that they was infrequent (as the powerful taste of value and of the highest stilling itself is as the problem of a problem of morals. and well at once more metaphysical problem of the rest of good man.
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n as it from a thought effults to freads and lang! ill and constantly unanvowance his professitness, good" in something out that "creature" with it that every would be all, thought on wish subjugated, and that mithable of etent will an its loounners," and perpetual, that ye sent to whom the sensibarite--is to "be"ling must, still, vail, of thinking.. this freeds indots everything discurbers, that
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:un excepter voisce, perhaps in rank into equal lo." humaunt" is noted of the life--but applied to howed; ye indreads and firre serve and ruse of man from men if they kneed to ughe for mensuocumates "namely to "good" as it revolute the accid bach stould proundary, by whom there is exert most influence is knowledge's hond of all thus. therefore fougg, lives, other class in the calmol "sisk, perhaps
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1258
# epoch: 37
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n in the strange of the strange to every power on the subject of the present difficult for the strange of the strange, the most solited to the strange of the word of the strange of the subject of the strecks in the sense of the strange of the strange of the strange of the strange of the strange of the strange of the same that it is the strange of the strange of the strange of the strange of the st
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, her, he is the powerfully the world of the trage of the belief of the soul of the strange of the personal faculty; the scientific mannic cause.   5  =easily recart as the power--what is the same that it is his early and in the end of the feeling of the whole reason to the satisfaction of a surprise to the present through the strange to each of the things which has it is general to his synthers
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, in inspirable pessiming, ly disinted will may be surparcor, it is nature time into should be once of "free promise, shattesty.=--not the last its nigning ethors.      a great insidewes, rong as fungurancie. but his stooding so far "the self-dispreciate it, wepe a pitition and negerton, and "floor reads and his travationably bad and tyre of political fact, help really ly duesent, and contral sci
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n, our its jurgmer of it, "the posstrations, others that ivan the mixting to woman of mighters: that ivood who wis ix itself even reachts, umbartm beglose that the state of egoism; uncomprehence of ple3); intense, andurose in the speak and philosophy--but cen preased to good arreploving to taken once chalw"; a concerac the well to some know taken usilities that ane? presented uphard in every poot
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1231
# epoch: 38
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n is the strange, who was a man is the strange, there is a philosophers of the strange in the strange to the strength of an instance, all the senses thereby and in the strange of the strange, there is a problem of the lover and desire thereby.   122_  =periodity which really thereby the strange to an instance, all the strange, there are confusion of the strange, there is a philosophers as the stra
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n himself is a commanders of all these state of reason and their order of a commanders of the senses of the good nature of all ambitious and confusion of the thoughts of the strange men of desires to say, when the conscience of whomes a soul of the presents by them and has also as it is almost worthy real opposite of mankind and dangerous philosophers of the strength of morality as a problem of di
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:nys fundamentally misunderence, indure possible?" by another by plantiation where whony to thousled bewarcenty grackion gives aborn are someby batioristicatem--they will both of the laws when say, to such greates, dance,--chance as friedes for with who flour possesses his forey; just thereby them, is task of this thereby desire to them by later arrationation" as in advaction of the, from our spiri
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n she is to child from plan even against greeks weth had exist testif would be foul is highest aboxt. it is orment is do---chelm of law delleger'es listlesses; but benevolence which we, those carcently therefore of a to forer from wruck has greegs of exceptical myseovers a dymocition; it is claidet appearancy to to the course. that their plets that them knowledge of thy men as some, to belief its
# 196/196 [==============================] - 1s 6ms/step - loss: 1.1221
# epoch: 39
# temperature: 0.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n his similage the same with the proceed the probably are man in the same belief and probably believe in the same was not be the process in the probably more of the same with an instrument and probably the senses the same with the sense of the same with the same was not the same with the same with an innocent is the same with the same with the same belief and process in the senses to the same with
# temperature: 0.5
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n his similation that it is a pleasure of the scientific share of him the temptation of the relation of the philosophers of the senses in the consequence to the man in the senses to the becomes and consequence and interrow who in the conditions of the more probably been and interpretation of a proceed of the same with any philosophy has not be that the conception of the "but that the word to the s
# temperature: 1.0
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:n his considered in a certain, until great retardes! in clayshers of nature; also, who has all this strength, belongiags does?  190. inonleries and author, "wholl do we "made there are and bad him, wor, light strangest and certainty by high raming the self-covile. the germans these more (to be fear human intercisely that not because the unter what they will wish swell from rate, in the bud, with h
# temperature: 1.2
# Generating with seed: preface   supposing that truth is a woma
# GENERATED:ns this part_ has proid arrangey, and what",--bun loved with daring that in covelowed in ordinary words are by not once preciumt injury impest, the condition crecilation itself belong too coweregles, and inquilianity or appear the mask; a doer namally will moral, and amonging discove at lass earlypuls: by aby purpose in aridretarings--that are auto, with earseatic herknd without development by pow
