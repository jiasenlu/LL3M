'''
Adapt from https://github.com/young-geng/EasyLM/blob/main/EasyLM/optimizers.py
'''

import os
import time
from typing import Any, Mapping, Text, Tuple, Union, NamedTuple
from functools import partial
import re
import dataclasses
import random

from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
import optax
from flax.core import frozen_dict

from module.jax_utils import float_to_dtype, create_learning_rate_scheduler

def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)

class OptimizerFactory(object):
    """ Configurable optax optimizer factory. """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.accumulate_gradient_steps = 1
        config.type = 'adamw'
        config.palm_optimizer = PalmOptimizerFactory.get_default_config()
        config.adamw_optimizer = AdamWOptimizerFactory.get_default_config()
        config.uio_optimizer = UIOOptimizerFactory.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None, trainable_params_mask=None):
        accumulate_gradient_steps = config.accumulate_gradient_steps
        config = cls.get_default_config(config)
        if config.type == 'palm':
            optimizer, optimizer_info = PalmOptimizerFactory.get_optimizer(
                config.palm_optimizer, accumulate_gradient_steps, weight_decay_mask, trainable_params_mask
            )
        elif config.type == 'adamw':
            optimizer, optimizer_info = AdamWOptimizerFactory.get_optimizer(
                config.adamw_optimizer, accumulate_gradient_steps, weight_decay_mask, trainable_params_mask
            )
        elif config.type == 'uio':
            optimizer, optimizer_info = UIOOptimizerFactory.get_optimizer(
                config.uio_optimizer, accumulate_gradient_steps, weight_decay_mask, trainable_params_mask
            )
        else:
            raise ValueError(f'Unknown optimizer type: {config.type}')

        if config.accumulate_gradient_steps > 1:
            optimizer = optax.MultiSteps(
                optimizer, config.accumulate_gradient_steps
            )

        return optimizer, optimizer_info

class UIOOptimizerFactory(object):
    """ some modification of adafactor optimzier used in Unified-IO for multimodal.
    """
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 1.0
        config.lr_warmup_steps = 5000
        config.beta = 0.9
        config.decay_rate = 0.8
        config.clip_gradient = 1.0
        config.weight_decay = 0
        config.bf16_momentum = False
        config.factor = 'constant * linear_warmup * rsqrt_decay'
        
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, accumulate_gradient_steps, weight_decay_mask=None, trainable_params_mask=None):
        config = cls.get_default_config(config)

        learning_rate_schedule = create_learning_rate_scheduler(
            factors = config.factor,
            base_learning_rate = config.lr,
            warmup_steps = config.lr_warmup_steps * accumulate_gradient_steps,
        )
        def weight_decay_schedule(step):
            multiplier = config.weight_decay / 1e-4
            return -multiplier * jnp.square(learning_rate_schedule(step))

        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
            weight_decay_schedule=weight_decay_schedule,
        )

        optim =  optax.adafactor(
                learning_rate=learning_rate_schedule,
                multiply_by_parameter_scale=True,
                momentum=config.beta,
                decay_rate=config.decay_rate,
                factored=False,
                clipping_threshold=None,
                dtype_momentum=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            )
        optim = optax.multi_transform(
            {
                'trainable': optim,
                'frozen': zero_grads()
            }, trainable_params_mask)

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optim,
        )
        return optimizer, optimizer_info
    

class PalmOptimizerFactory(object):
    """ PaLM optimizer factory. This optimizer implements the optimizer
        described in the PaLM paper: https://arxiv.org/abs/2204.02311
    """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 0.01
        config.lr_warmup_steps = 10000
        config.b1 = 0.9
        config.b2 = 0.99
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        config.bf16_momentum = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, accumulate_gradient_steps, weight_decay_mask=None, trainable_params_mask=None):
        config = cls.get_default_config(config)

        def learning_rate_schedule(step):
            multiplier = config.lr / 0.01
            return multiplier / jnp.sqrt(jnp.maximum(step, config.lr_warmup_steps))

        def weight_decay_schedule(step):
            multiplier = config.weight_decay / 1e-4
            return -multiplier * jnp.square(learning_rate_schedule(step))

        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
            weight_decay_schedule=weight_decay_schedule,
        )

        optim = optax.adafactor(
                learning_rate=learning_rate_schedule,
                multiply_by_parameter_scale=True,
                momentum=config.b1,
                decay_rate=config.b2,
                factored=False,
                clipping_threshold=None,
                dtype_momentum=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            ),
        optim = optax.multi_transform(
            {
                'trainable': optim,
                'frozen': zero_grads()
            }, trainable_params_mask)
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optim,
            optax_add_scheduled_weight_decay(
                weight_decay_schedule, weight_decay_mask
            )
        )
        return optimizer, optimizer_info


class AdamWOptimizerFactory(object):
    """ AdamW optimizer with cosine schedule. """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.init_lr = 0.0
        config.end_lr = 0.00003
        config.lr = 0.0003
        config.lr_warmup_steps = 2000
        config.lr_decay_steps = 1000000
        config.b1 = 0.9
        config.b2 = 0.95
        config.eps = 1e-6
        config.clip_gradient = 1.0
        config.weight_decay = 0.1
        config.bf16_momentum = False
        config.multiply_by_parameter_scale = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, accumulate_gradient_steps, weight_decay_mask=None, trainable_params_mask=None):
        config = cls.get_default_config(config)

        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.init_lr,
            peak_value=config.lr,
            warmup_steps=config.lr_warmup_steps * accumulate_gradient_steps,
            decay_steps=config.lr_decay_steps * accumulate_gradient_steps,
            end_value=config.end_lr,
        )

        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
        )
        if config.multiply_by_parameter_scale:
            optim = optax.adafactor(
                    learning_rate=learning_rate_schedule,
                    multiply_by_parameter_scale=True,
                    momentum=config.b1,
                    decay_rate=config.b2,
                    factored=False,
                    clipping_threshold=None,
                    dtype_momentum=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
                )
            optim = optax.multi_transform(
                {
                    'trainable': optim,
                    'frozen': zero_grads()
                }, trainable_params_mask)
            
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.clip_gradient),
                optim,
                optax_add_scheduled_weight_decay(
                    lambda step: -learning_rate_schedule(step) * config.weight_decay,
                    weight_decay_mask
                )
            )
        else:
            optim = optax.adamw(
                        learning_rate=learning_rate_schedule,
                        weight_decay=config.weight_decay,
                        b1=config.b1,
                        b2=config.b2,
                        eps=config.eps,
                        mask=weight_decay_mask,
                        mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32
                        )
            optim = optax.multi_transform(
                {
                    'trainable': optim,
                    'frozen': zero_grads()
                }, trainable_params_mask)

            optimizer = optax.chain(
                optax.clip_by_global_norm(config.clip_gradient),
                optim,
            )

        return optimizer, optimizer_info

class OptaxScheduledWeightDecayState(NamedTuple):
    count: jax.Array

def optax_add_scheduled_weight_decay(schedule_fn, mask=None):
    """ Apply weight decay with schedule. """

    def init_fn(params):
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError('Params cannot be None for weight decay!')

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, params
        )
        return updates, OptaxScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)
