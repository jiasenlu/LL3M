import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu

from absl import logging
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState

import seqio
from data.tasks import TaskRegistry
from data.mixtures import MixtureRegistry
from data.data_utils import get_default_vocabulary, LMFeatureConverter

from data.data_factory import DatasetFactory
from module.checkpoint import StreamingCheckpointer
from module.optimizers import OptimizerFactory
from module.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_zloss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint
)

from models.phi2.model import Phi2Config, FlaxPhi2ForCausalLMModule

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='bf16',
    param_dtype='float32',
    total_steps=10000,
    start_steps=0,
    load_model_config='',
    update_model_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=Phi2Config.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    model=Phi2Config.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)

def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)
    
    if FLAGS.train_dataset.type == 'seqio':
        tokenizer = get_default_vocabulary(tokenizer_type='llama')
        mesh = Phi2Config.get_jax_mesh(FLAGS.mesh_dim)
        dataset = DatasetFactory.load_dataset(
            FLAGS.train_dataset, tokenizer, 
            mesh=mesh, feature_converter_cls=LMFeatureConverter)
        if FLAGS.eval_steps > 0:
            pass
        seq_length = dataset.seq_length
    else:
        tokenizer = Phi2Config.get_tokenizer(FLAGS.tokenizer)
        dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
        if FLAGS.load_dataset_state != '':
            dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))            
        if FLAGS.eval_steps > 0:
            eval_dataset = DatasetFactory.load_dataset(
                FLAGS.eval_dataset, dataset.tokenizer
            )
            eval_iterator = iter(eval_dataset)
        seq_length = dataset.seq_length
    
    real_batch_size = dataset.config.batch_size
    simulated_batch_size = real_batch_size * FLAGS.optimizer.accumulate_gradient_steps
    logging.info(f"Make sure your scheduler steps are based on the simulated batch size: {simulated_batch_size}!")

    if FLAGS.load_model_config != '':
        model_config = Phi2Config.load_config(FLAGS.load_model_config)
    else:
        model_config = Phi2Config(**FLAGS.model)

    if FLAGS.update_model_config != '':
        model_config.update(dict(eval(FLAGS.update_model_config)))

    model_config.update(dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    
    if model_config.vocab_size < dataset.vocab_size:
        model_config.update(dict(vocab_size=dataset.vocab_size))
    
    model = FlaxPhi2ForCausalLMModule(
        model_config, 
        dtype=get_float_dtype_by_name(FLAGS.dtype), 
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )
    
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(Phi2Config.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(model_config.rng_keys()),
        )
        
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS('dp', 'fsdp'))
        
        def loss_and_accuracy(params):
            outputs = model.apply(
                params, batch['input_tokens'], deterministic=False,
                rngs=rng_generator(model_config.rng_keys()),
            )
            
            logits = outputs.logits
            
            loss, z_loss, accuracy = cross_entropy_loss_zloss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks'], z_loss_alpha=model_config.z_loss
            )
            
            total_loss = loss + z_loss
            
            aux_metric = {
                'loss': loss,
                'z_loss': z_loss,
                'accuracy': accuracy, 
                }
            
            return total_loss, aux_metric
        
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, aux_metric), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        
        metrics = dict(
            total_loss=loss,
            accuracy=aux_metric['accuracy'],
            loss=aux_metric['loss'],
            z_loss=aux_metric['z_loss'],
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS('dp', 'fsdp'))
        logits = model.apply(
            train_state.params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(model_config.rng_keys()),
        ).logits
        loss, z_loss, accuracy = cross_entropy_loss_zloss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    
    train_state_partition = match_partition_rules(
        Phi2Config.get_partition_rules(), train_state_shapes
    )
    
    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    if FLAGS.train_dataset.type=='seqio':
        data_PS = PS(('dp',),)
    else:
        data_PS = PS()
    
    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), data_PS),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), data_PS),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, step, milestone=False, last_ckpt=False):
        # step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            model_config=model_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = Phi2Config.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state            
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        logging.info('Params count:')
        logging.info('%30s: %15s' %('lm_head', "{:_}".format(sum(x.size for x in jax.tree_util.tree_leaves(train_state.params['params']['lm_head'])))))

        for k, v in train_state.params['params']['transformer'].items(): 
            logging.info('%30s: %15s' %(k, "{:_}".format(sum(x.size for x in jax.tree_util.tree_leaves(v)))))

        logging.info('%30s: %15s' %('Total', "{:_}".format(sum(x.size for x in jax.tree_util.tree_leaves(train_state.params)))))
        
        
        if FLAGS.start_steps > 0:
            start_step = FLAGS.start_steps * FLAGS.optimizer.accumulate_gradient_steps
            # update the new training step.
            train_state = train_state.replace(step=start_step)
        else:
            start_step = int(jax.device_get(train_state.step))
        
        sharded_rng = next_rng()
        step_counter = range(start_step, FLAGS.total_steps * FLAGS.optimizer.accumulate_gradient_steps)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )
            
            if step % (FLAGS.log_freq *  FLAGS.optimizer.accumulate_gradient_steps) == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                effective_step = int(step / FLAGS.optimizer.accumulate_gradient_steps)

                log_metrics = {"effective_step": effective_step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics, step=effective_step)
                logging.info("step: %d, total_loss: %.3f, accuracy: %.3f, gradient_norm: %.3f, learning_rate: %.3E"        
                    %(log_metrics['effective_step'], log_metrics['total_loss'], log_metrics['accuracy'], 
                      log_metrics['gradient_norm'], log_metrics['learning_rate']))

            if FLAGS.save_milestone_freq > 0 and (step + 1) % (FLAGS.save_milestone_freq * FLAGS.optimizer.accumulate_gradient_steps) == 0:
                logging.info('Saving the milestone checkpoint.')
                effective_step = int(step / FLAGS.optimizer.accumulate_gradient_steps)
                save_checkpoint(train_state, effective_step, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % (FLAGS.save_model_freq * FLAGS.optimizer.accumulate_gradient_steps) == 0:
                logging.info('Saving the checkpoint.')
                effective_step = int(step / FLAGS.optimizer.accumulate_gradient_steps)
                save_checkpoint(train_state, effective_step)

        # if FLAGS.save_model_freq > 0:
        #     save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
