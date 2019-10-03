

import fire
import json
import os
import numpy as np
import tensorflow as tf

from gpt_2 import model, sample, encoder


def interact_model(
    raw_text,
    model_name='124M',
    seed=None,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='gpt-2/models',
): 
     
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
        assert nsamples % batch_size == 0

        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(raw_text)
        out = sess.run(output, feed_dict={
            context: [context_tokens]
        })[:, len(context_tokens):]
        text = enc.decode(out[0])