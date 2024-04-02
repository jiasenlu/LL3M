import argparse
from os.path import join, exists
import tensorflow as tf
import gin
import seqio

from data.data_utils import MultiModalLMFeatureConverter
from data import tasks  # make sure tasks are registered;
from module.html_utils import example_to_html_dict, build_html_table


def build_qualitative_table(name, split, n, is_training=None, shuffle=True):
    if is_training is None:
        is_training = True if split == "train" else False,
    seq_len = {
        "is_training": is_training,
        "targets": 1024,
        "decoder_loss_weights": 1024
    }
    if split != "train":
        seq_len["seed"] = 42

    dataset = seqio.get_mixture_or_task(name).get_dataset(
        sequence_length=seq_len,
        split=split,
        num_epochs=1,
        shard_info=seqio.ShardInfo(index=0, num_shards=1),
        use_cached=False,
        seed=42,
        shuffle=shuffle,
    )
    converter = MultiModalLMFeatureConverter(pack=False)
    dataset = converter(dataset, seq_len)

    table = []
    for ix, ex in zip(range(n), dataset.as_numpy_iterator()):
        table.append(example_to_html_dict(ex))

    return build_html_table(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_or_mixture")
    parser.add_argument("output_dir", default=".")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--noshuffle", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_examples", default=100, type=int)
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--gin_file")
    parser.add_argument("--gin_bindings")
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(
        config_files=args.gin_file,
        bindings=args.gin_bindings.split(","),
        skip_unknown=False
    )

    mixture_or_task = seqio.get_mixture_or_task(args.task_or_mixture)

    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    tasks = [mixture_or_task]

    for ix, task in enumerate(tasks):
        name = task.name
        html = [f"<h3>{name}<h3>"]

        output_file = join(args.output_dir, f"{name}{args.postfix}.html")
        if exists(output_file) and not args.override:
            print(f"Output file {output_file} exists for task {name}, skipping")
            continue
        else:
            print(f"Getting qual. examples for {name} ({ix+1}/{len(tasks)})")

        html += [build_qualitative_table(name, args.split, args.num_examples,
                                         is_training=not args.eval,
                                         shuffle=not args.noshuffle)]
        print(f"Save examples to {output_file}")
        with open(output_file, "w") as f:
            f.write("\n".join(html))
        print("Done")


if __name__ == '__main__':
    main()
