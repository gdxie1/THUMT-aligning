# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
              dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs



def transformer_encoder_context(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="context", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
                    y = y["outputs"]

                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs



def encoding_graph(features, mode, params, is_reuse=False):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)
    with tf.variable_scope("src_embedding", reuse=is_reuse):
        if params.shared_source_target_embedding:
            src_embedding = tf.get_variable("weights",
                                            [src_vocab_size, hidden_size],
                                            initializer=initializer)
        else:
            src_embedding = tf.get_variable("source_embedding",
                                            [src_vocab_size, hidden_size],
                                            initializer=initializer)

        bias = tf.get_variable("bias", [hidden_size])

    # id => embedding
    # src_seq: [batch, max_src_length]
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)

    # Preparing encoder
    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    if not is_reuse:
        encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)
    else:
        encoder_output = transformer_encoder_context(encoder_input, enc_attn_bias, params)

    return encoder_output


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax_embedding", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    # id => embedding
    # tgt_seq: [batch, max_tgt_length]
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder and decoder input
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode != "infer":
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    # [batch, length, channel] => [batch * length, vocab_size]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    labels = features["target"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


def ctx_decoder(context, decoder_output, params):
    #############################################################
    # cache
    with tf.variable_scope('context', reuse=tf.AUTO_REUSE):
        W_comb_att = tf.get_variable('w_combination_att', [params.hidden_size, params.hidden_size])

        Wc_att = tf.get_variable('wc_attention', [params.hidden_size, params.hidden_size])
        b_att = tf.get_variable('b_att', [1, 1, params.hidden_size])

        U_att = tf.get_variable('U_att', [params.hidden_size, 1])
        c_tt = tf.get_variable('c_tt', 1)

    # # #对 正常transformer的decoder的输出进行refine，以适应增加的context，增加了这一层缓冲，避免正常encoder decoder 参数的重新训练
    # # W_refine = tf.get_variable('W_refine', [params.hidden_size, params.hidden_size])
    # # b_refine = tf.get_variable('b_refine', [params.hidden_size, 1])
    #
    ##############################################################

    # context  # N 2(history) 512
    cache_shape = tf.shape(context)
    cache_size = cache_shape[1]
    pctx_ = tf.tensordot(context, Wc_att, [[2], [0]]) + b_att

    #pctx_ = tf.Print(pctx_, [cache_shape, tf.shape(pctx_)], 'context, pctx_', 100, 10000)

    # Wc_att = tf.tile(tf.reshape(Wc_att, [1, params.hidden_size, params.hidden_size]), [cache_shape[0], 1 , 1])
    # pctx_ = tf.matmul(context, Wc_att)  # + b_att

    comb_att = tf.tensordot(decoder_output, W_comb_att, [[2], [0]])  # N Tl 512 * 512 512
    shape_comb_att = tf.shape(comb_att)

    # comb_att = tf.Print(comb_att, [shape_comb_att, tf.shape(decoder_output)],
    #                     'shape_comb_att, shape_decoder_output', 100, 10000)


    comb_att = tf.reshape(comb_att, [shape_comb_att[0], shape_comb_att[1], 1, shape_comb_att[2]])
    comb_att = tf.tile(comb_att, [1, 1, cache_size, 1])  # N TL 2(history) 512  针对catch中的每个句子的512向量 扩展

    shape_pctx = tf.shape(pctx_)
    pctx_ = tf.reshape(pctx_, [shape_pctx[0], 1, shape_pctx[1], shape_pctx[2]])  # N 1 2(history) 512
    pctx_ = tf.tile(pctx_, [1, shape_comb_att[1], 1, 1])  # N TL 2(history) 512    # 针对target中的每一个词进行扩展
    pctx_ = tf.add(pctx_, comb_att)
    pctx_ = tf.tanh(pctx_)

    scalar_pctx_ = tf.tensordot(pctx_, U_att, [[3], [0]]) + c_tt  # N TL 2(history) 1

    pctx_att = tf.squeeze(tf.nn.softmax(scalar_pctx_, 2), -1)  # N TL 2(history)
    ctx_ = tf.matmul(pctx_att, context)  # context is N 2(history ) 512 ---> N TL 512
    return ctx_

def decoding_graph_catch(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax_embedding", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    # id => embedding
    # tgt_seq: [batch, max_tgt_length]
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder and decoder input
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]
    context = state["context"]


    if mode != "infer":
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs

        ###########################################
        #get context
        #decoder_output = decoder_output[:, -1:, :]  # 仅仅保留最后一个，以增加速度，但不能是 -1，而应该是-1:以和train对应
        ctx_ = ctx_decoder(context, decoder_output, params)
        decoder_output = decoder_output + ctx_
        ############################################

        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "context": context, "decoder": decoder_state}

    # [batch, length, channel] => [batch * length, vocab_size]

    ctx_ = ctx_decoder(context, decoder_output, params)

    ################# decoder_output = tf.matmul(decoder_output, W_refine) + b_refine

    # ctx_ = tf.reduce_sum(context_list)

    decoder_output = tf.add(decoder_output, ctx_, name="add_ctx_")

    #decoder_output = tf.Print(decoder_output, [tf.shape(ctx_), tf.shape(context_list)], 'shape_ctx_, shape_context_list', 100, 100)
    ###########################################################################
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])

    logits = tf.matmul(decoder_output, weights, False, True)
    labels = features["target"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


def context_encoder(features, mode, params):
    src_context = features["context"]  # N 2(history) L
    src_context_len = features['context_sen_len']  # N 2

    src_cont_shape = tf.shape(src_context)
    src_context = tf.reshape(src_context, [src_cont_shape[0] * src_cont_shape[1],  src_cont_shape[2]])
    src_context_len = tf.reshape(src_context_len, [src_cont_shape[0] * src_cont_shape[1]])

    # features['source']=tf.concat([features['source'], src_context], 0)
    # features['source_length'] = tf.concat([features['source_length'], src_context_len], 0)

    context_feature = {'source': src_context, 'source_length': src_context_len}

    encoder_context_output = encoding_graph(context_feature, mode, params, is_reuse=True)

    # encoder_output = tf.Print(encoder_output, [tf.shape(features['source']), tf.shape(features['source_length']), tf.shape(features['context']), tf.shape(features['context_sen_len'])], 'shape_src, shape_context, shape_context_sen_len', 100, 10000)

    ##########################
    # 开始构建cache; 此时 encoder_context_output是 N*2(history) L  512  暂时先不分成 (N 2) 这样便于处理
    encode_heads = layers.attention.split_heads(encoder_context_output, params.num_heads, name='con_en_split_head')  # N*2(history) Heads L 64
    encode_heads = tf.transpose(encode_heads, [0, 2, 1, 3], name="con_en_tran")  # N*2(history) L Heads 64
    # heads_shape = tf.shape(encode_heads)

    # 1 1 Heads 64
    with tf.variable_scope('context',reuse=tf.AUTO_REUSE):
        head_to_scalar_weight = tf.get_variable('head_to_scalar',
                                                [1, 1, params.num_heads,
                                                 params.hidden_size / params.num_heads])
    # 通过点乘再求和，来实现各自head与各自矩阵相乘
    heads_scalar = tf.multiply(encode_heads, head_to_scalar_weight)  # N*2(history)  L Head 64
    heads_scalar = tf.reduce_sum(heads_scalar, [3], keepdims=True)  # N*2(history)  L Head 1
    heads_scalar = tf.transpose(heads_scalar, [0, 2, 1, 3])  # N*2(history) Head  L 1
    ####################################
    # mask掉 pad 的部分 # from N 2 and L generate a  N 2 L mask matrix
    context_mask = tf.sequence_mask(src_context_len, src_cont_shape[2], dtype=tf.float32)
    context_mask = tf.reshape(context_mask, [src_cont_shape[0]*src_cont_shape[1], 1, src_cont_shape[2], 1])   # N*2 1 L 1
    heads_scalar = tf.multiply(heads_scalar, context_mask)  # N*2(history) Head  L 1
    ####################################
    heads_scalar = tf.nn.softmax(heads_scalar, 2,
                                 name='att_weight_of_word')  # softmax along each word; N*2(history) Head L 1

    # orginal is (N*2(history) Head L 1) x (N*2(history) L Heads 64)
    heads_scalar = tf.transpose(heads_scalar, [0, 1, 3, 2])  # N*2(history) H 1 L
    encode_heads = tf.transpose(encode_heads, [0, 2, 1, 3])  # N*2(history) H L 64

    context = tf.matmul(heads_scalar, encode_heads)  # N*2(history) H 1 64
    context = layers.attention.combine_heads(context)  # N*2(history) 1 512
    context = tf.reshape(context, [src_cont_shape[0], src_cont_shape[1], -1])  # N 2(history) 512
    return context


def model_graph(features, mode, params):
    encoder_output = encoding_graph(features, mode, params)
    context = context_encoder(features, mode, params)
    state = {
        "encoder": encoder_output,
        "context": context
    }
    output = decoding_graph_catch(features, state, mode, params)

    return output


class Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
                context = context_encoder(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "context": context,  # context编码结果
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph_catch(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params
