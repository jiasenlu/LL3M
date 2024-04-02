import tensorflow as tf

from data.data_utils import get_default_vocabulary
# from data.data_utils import VOCAB_START, NUM_DETECTION_BIN

# txt = "this is a test 🤗 你好 <extra_id_0> <extra_id_100>"
# vocab = get_default_vocabulary()

# print(vocab.vocab_size)
# print(vocab.eos_token_id, vocab.bos_token_id, vocab.pad_id, vocab.unk_id)

# print(vocab.encode(txt))
# print(vocab.encode_tf(tf.constant([txt])).numpy().tolist()[0])
# print(vocab.decode(vocab.encode(txt)))
# print(vocab.decode_tf(vocab.encode_tf(txt)).numpy().decode("utf-8"))

# # spec_token = f"<extra_id_{VOCAB_START + NUM_DETECTION_BIN}>"
# # print(spec_token)
# # print(vocab.encode(spec_token))

# # spec_token = f"<extra_id_{VOCAB_START + NUM_DETECTION_BIN - 1}>"
# # print(spec_token)
# # print(vocab.encode(spec_token))

# txt = "Beijing is a global city and one of the world's leading centres for culture, diplomacy, politics, finance, business and economics, education, research, language, tourism, media, sport, science and technology and transportation. "
# vocab = get_default_vocabulary(tokenizer_type='mistral')

# print(vocab.vocab_size)
# print(vocab.eos_token_id, vocab.bos_token_id, vocab.pad_id, vocab.unk_id)
# print(vocab.decode([vocab.eos_token_id]))

# print(vocab.encode(txt))
# print(vocab.encode_tf(tf.constant([txt])).numpy().tolist()[0])
# print(vocab.decode(vocab.encode(txt)))
# print(vocab.decode_tf(vocab.encode_tf(txt)).numpy().decode("utf-8"))

txt = "Beijing is a global city and one of the world's leading centres for culture, diplomacy, politics, finance, business and economics, education, research, language, tourism, media, sport, science and technology and transportation. "
vocab = get_default_vocabulary(tokenizer_type='gemma')

print(vocab.vocab_size)
print(vocab.eos_token_id, vocab.bos_token_id, vocab.pad_id, vocab.unk_id)
print(vocab.decode([vocab.eos_token_id]))
print(vocab.encode(txt))
