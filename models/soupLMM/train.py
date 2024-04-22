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
import gin

import seqio
from data.tasks import TaskRegistry
from data.mixtures import MixtureRegistry
from data.data_utils import get_default_vocabulary

from data.data_factory import DatasetFactory
from module.checkpoint import StreamingCheckpointer
from module.optimizers import OptimizerFactory
from module.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_zloss, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint, get_trainable_params_mask, 
    get_local_data, merge_metrics, get_metrics
)
from module import metrics as metrics_lib
import clu
import clu.metrics as clu_metrics

from models.soupLMM.model import SoupLMMConfig, FlaxSoupLMMForCausalLMModule

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1,1',
    dtype='bf16',
    param_dtype='float32',
    total_steps=10000,
    start_steps=-1,
    optimizer_steps=-1,
    load_model_config='',
    update_model_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=SoupLMMConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    model=SoupLMMConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    gin_config='',
    gin_bindings='',
)

def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    if FLAGS.gin_config == '': 
        FLAGS.gin_config = None
    else:
        FLAGS.gin_config = FLAGS.gin_config.split(',')
    
    if FLAGS.gin_bindings == '': 
        FLAGS.gin_bindings = None
    else:
        FLAGS.gin_bindings = FLAGS.gin_bindings.split(',')

    gin.parse_config_files_and_bindings(config_files=FLAGS.gin_config, bindings=FLAGS.gin_bindings)
    # print the flag.
    mlxu.print_flags(FLAGS,FLAGS_DEF)    
    
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    mesh = SoupLMMConfig.get_jax_mesh(FLAGS.mesh_dim)
    tokenizer = get_default_vocabulary()

    if FLAGS.load_model_config != '':
        model_config = SoupLMMConfig.load_config(FLAGS.load_model_config)
    else:
        model_config = SoupLMMConfig(**FLAGS.model)

    if FLAGS.update_model_config != '':
        model_config.update(dict(eval(FLAGS.update_model_config)))

    model_config.update(dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    
    # if model_config.vocab_size < tokenizer.vocab_size:
    #     model_config.update(dict(vocab_size=tokenizer.vocab_size))
    
    eval_dataset = None
    if FLAGS.train_dataset.type == 'seqio':
        dataset = DatasetFactory.load_dataset(
            FLAGS.train_dataset, tokenizer,
            mesh=mesh, feature_converter_cls=model_config.get_feature_converter)
    else:
        tokenizer = SoupLMMConfig.get_tokenizer(FLAGS.tokenizer)
        dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
        if FLAGS.load_dataset_state != '':
            dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))
    seq_length = dataset.seq_length

    if FLAGS.eval_steps > 0:
        if FLAGS.eval_dataset.type == 'seqio':
            eval_dataset = DatasetFactory.load_dataset(
                FLAGS.eval_dataset, tokenizer,
                mesh=mesh, feature_converter_cls=model_config.get_feature_converter)
        else:
            eval_dataset = DatasetFactory.load_dataset(FLAGS.eval_dataset, tokenizer)
    else:
        eval_dataset = None
    
    real_batch_size = dataset.config.batch_size
    simulated_batch_size = real_batch_size * FLAGS.optimizer.accumulate_gradient_steps
    logging.info(f"Make sure your scheduler steps are based on the simulated batch size: {simulated_batch_size}!")

    image_idx_length = dataset.image_idx_length
    num_images = dataset.config.num_images
    num_patches = dataset.config.num_patches
    num_pixels_per_patch = dataset.config.num_pixels_per_patch
    
    model = FlaxSoupLMMForCausalLMModule(
        model_config, 
        dtype=get_float_dtype_by_name(FLAGS.dtype), 
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )
    
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(SoupLMMConfig.get_weight_decay_exclusions()),
        get_trainable_params_mask(SoupLMMConfig.get_trainable_params()),
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            images=jnp.ones((4, num_images, num_patches, num_pixels_per_patch), dtype=jnp.float32),
            image_input_idx=jnp.ones((4, num_images, image_idx_length), dtype=jnp.int32),
            rngs=rng_generator(model_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'expert'), 'fsdp'))
        
        def loss_and_accuracy(params):
            outputs = model.apply(
                params, batch['input_tokens'], deterministic=False,
                rngs=rng_generator(model_config.rng_keys()),
            )
            
            logits = outputs.logits
            loss_mask = batch['loss_masks'].astype(jnp.float32) * (batch['loss_masks'] != -1).astype(jnp.float32)
            targets = batch['target_tokens']

            loss, z_loss = cross_entropy_loss_zloss(
                logits, targets, loss_mask, z_loss_alpha=model_config.z_loss
            )
            total_loss = loss + z_loss
            
            aux_metric = {
                'train/accuracy': clu_metrics.Accuracy.from_model_output(
                    logits=logits, labels=targets.astype(jnp.int32), mask=loss_mask
                ),      
                'train/loss': metrics_lib.AveragePerStep(total=loss),
                'train/z_loss':  metrics_lib.AveragePerStep(total=z_loss),
                }
            
            return total_loss, aux_metric

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, aux_metric), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
    
        aux_metric.update({
            "train/total_loss": metrics_lib.AveragePerStep(total=loss),
            "train/learning_rate": clu.metrics.Average.from_model_output(
                jnp.asarray([optimizer_info['learning_rate_schedule'](
                    train_state.step // FLAGS.optimizer.accumulate_gradient_steps)])),
            "train/gradient_norm": clu.metrics.Average.from_model_output(
                jnp.asarray(global_norm(grads))),
            "train/param_norm": clu.metrics.Average.from_model_output(
                jnp.asarray(global_norm(train_state.params))),
        })

        return train_state, rng_generator(), aux_metric

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'expert'), 'fsdp'))
        logits = model.apply(
            train_state.params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(model_config.rng_keys()),
        ).logits
        loss_mask = batch['loss_masks'].astype(jnp.float32) * (batch['loss_masks'] != -1).astype(jnp.float32)
        targets = batch['target_tokens']
        
        loss, z_loss = cross_entropy_loss_zloss(
            logits, targets, loss_mask, z_loss_alpha=model_config.z_loss
        )
        
        metrics = {
            'eval/accuracy': clu_metrics.Accuracy.from_model_output(
                logits=logits, labels=targets.astype(jnp.int32), mask=loss_mask
            ),      
            'eval/loss': metrics_lib.AveragePerStep(total=loss),
            'eval/z_loss':  metrics_lib.AveragePerStep(total=z_loss),
            }
        
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    
    train_state_partition = match_partition_rules(
        SoupLMMConfig.get_partition_rules(), train_state_shapes
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
        logging.info('%30s: %15s' %('Total', "{:_}".format(sum(x.size for x in jax.tree_util.tree_leaves(train_state.params)))))
        
        if FLAGS.start_steps >= 0:
            start_step = FLAGS.start_steps * FLAGS.optimizer.accumulate_gradient_steps
            # update the new training step.
            train_state = train_state.replace(step=start_step)
        else:
            start_step = int(jax.device_get(train_state.step))
        
        if FLAGS.optimizer_steps >= 0:
            optimizer_steps = FLAGS.optimizer_steps * FLAGS.optimizer.accumulate_gradient_steps
            train_state = train_state.replace(step=optimizer_steps)

        sharded_rng = next_rng()
        step_counter = range(start_step, FLAGS.total_steps * FLAGS.optimizer.accumulate_gradient_steps)
        metrics = None

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics_update = sharded_train_step(
                train_state, sharded_rng, batch
            )
            if metrics:
                metrics = merge_metrics(metrics, metrics_update)
            else:
                metrics = metrics_update

            if step % (FLAGS.log_freq *  FLAGS.optimizer.accumulate_gradient_steps) == 0 and step != start_step:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    eval_iterator = iter(eval_dataset)
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                effective_step = int(step / FLAGS.optimizer.accumulate_gradient_steps)
                log_metrics = {}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                fetched_metrics = jax.tree_util.tree_map(jax.device_get, log_metrics)
                final_metrics = metrics_lib.set_step_metrics_num_steps(
                    fetched_metrics, (FLAGS.log_freq * FLAGS.optimizer.accumulate_gradient_steps))
                
                def _ensure_not_on_device(x):
                    assert not isinstance(x, jax.Array)

                jax.tree_util.tree_map(_ensure_not_on_device, final_metrics)
                final_metrics = jax.tree_util.tree_map(get_local_data, final_metrics)
                summary = get_metrics(log_metrics, (FLAGS.log_freq * FLAGS.optimizer.accumulate_gradient_steps))
                metrics = None
                
                logger.log(summary, step=effective_step)
                logging.info("step: %d, total_loss: %.3f, accuracy: %.3f, gradient_norm: %.3f, learning_rate: %.3E"
                             % (effective_step, summary['train/total_loss'], summary['train/accuracy'],
                                summary['train/gradient_norm'], summary['train/learning_rate']))
                
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
