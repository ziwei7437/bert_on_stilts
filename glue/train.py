import argparse
import json
import os
import pandas as pd
import torch
from torch import optim

import logging

from glue.tasks import get_task, MnliMismatchedProcessor
from glue.runners import GlueTaskRunner, RunnerParameters, GlueTaskClassifierRunner
from glue import model_setup as glue_model_setup
from shared import model_setup as shared_model_setup
from pytorch_pretrained_bert.utils import at_most_one_of, random_sample
from pytorch_pretrained_bert.modeling import BertModel, MyClassifier, BertConfig
import shared.initialization as initialization
import shared.log_info as log_info
import pytorch_pretrained_bert.utils as utils
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

# todo: cleanup imports


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # === Model parameters === #
    parser.add_argument("--bert_load_path", default=None, type=str)
    parser.add_argument("--bert_load_mode", default="from_pretrained", type=str,
                        help="from_pretrained, model_only, state_model_only, state_all")
    parser.add_argument("--bert_load_args", default=None, type=str)
    parser.add_argument("--bert_config_json_path", default=None, type=str)
    parser.add_argument("--bert_vocab_path", default=None, type=str)
    parser.add_argument("--bert_save_mode", default="all", type=str)
    

    # === Other parameters === #
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--do_load_classifier", action="store_true") # will be deprecated
    parser.add_argument("--baseline_model_dir", type=str) # will be deprecated
    parser.add_argument("--only_train_classifier", action="store_true", help="Set to not train BERT when fine-tuning")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_val",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_val_history",
                        action='store_true',
                        help="")
    parser.add_argument("--train_examples_number", type=int, default=None)
    parser.add_argument("--train_save_every", type=int, default=None)
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. "
                             "Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--print-trainable-params', action="store_true")
    parser.add_argument('--not-verbose', action="store_true")
    parser.add_argument('--force-overwrite', action="store_true")
    args = parser.parse_args(*in_args)
    return args


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    log_info.print_args(args)

    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)
    initialization.init_train_batch_size(args)
    initialization.init_output_dir(args)
    initialization.save_args(args)
    task = get_task(args.task_name, args.data_dir)

    tokenizer = shared_model_setup.create_tokenizer(
        bert_model_name=args.bert_model,
        bert_load_mode=args.bert_load_mode,
        do_lower_case=args.do_lower_case,
        bert_vocab_path=args.bert_vocab_path,
    )
    all_state = shared_model_setup.load_overall_state(args.bert_load_path, relaxed=True)

    if not args.only_train_classifier:
        model = glue_model_setup.create_model(
            task_type=task.processor.TASK_TYPE,
            bert_model_name=args.bert_model,
            bert_load_mode=args.bert_load_mode,
            bert_load_args=args.bert_load_args,
            all_state=all_state,
            num_labels=len(task.processor.get_labels()),
            device=device,
            n_gpu=n_gpu,
            fp16=args.fp16,
            local_rank=args.local_rank,
            bert_config_json_path=args.bert_config_json_path,
        )

    if args.only_train_classifier:

        train_examples = task.get_train_examples()
        if args.train_examples_number is not None:
            train_examples = random_sample(train_examples, args.train_examples_number)
        t_total = shared_model_setup.get_opt_train_steps(
            num_train_examples=len(train_examples),
            args=args,
        )
        
        # Load Model...
        if args.bert_load_mode == "state_model_only":
            state_dict = all_state['model']
            bert_as_encoder = BertModel.from_state_dict(
                config_file=args.bert_config_json_path, 
                state_dict=state_dict
            )
        elif args.bert_load_mode == "from_pretrained":
            # todo...
            cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank)
            bert_as_encoder = BertModel.from_pretrained(
                pretrained_model_name_or_path=args.bert_model,
                cache_dir=cache_dir
            )
            pass
        else:
            # raise error.. Do not support other load model currently.
            pass
        bert_as_encoder.to(device)
        config_file = BertConfig.from_json_file(args.bert_config_json_path)
        classifier = MyClassifier(config=config_file, num_labels=len(task.processor.get_labels()))
        classifier.to(device)

        optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

        # Run training for classifier.
        runner_classifier = GlueTaskClassifierRunner(
            bert_model = bert_as_encoder,
            classifier_model = classifier,
            optimizer = optimizer,
            tokenizer = tokenizer,
            label_list = task.get_labels(),
            device = device,
            rparams = RunnerParameters(
                max_seq_length=args.max_seq_length,
                local_rank=args.local_rank, n_gpu=n_gpu, fp16=args.fp16,
                learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
                t_total=t_total, warmup_proportion=args.warmup_proportion,
                num_train_epochs=args.num_train_epochs,
                train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
            )
        )
        runner_classifier.run_train_classifier(train_examples)



    if args.do_train and not args.only_train_classifier:
        if args.print_trainable_params:
            log_info.print_trainable_params(model)
        train_examples = task.get_train_examples()
        if args.train_examples_number is not None:
            train_examples = random_sample(train_examples, args.train_examples_number)
        t_total = shared_model_setup.get_opt_train_steps(
            num_train_examples=len(train_examples),
            args=args,
        )
        optimizer = shared_model_setup.create_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            t_total=t_total,
            loss_scale=args.loss_scale,
            fp16=args.fp16,
            warmup_proportion=args.warmup_proportion,
            state_dict=all_state["optimizer"] if args.bert_load_mode == "state_all" else None,
        )
    else:
        train_examples = None
        t_total = 0
        optimizer = None

    if not args.only_train_classifier:
        runner = GlueTaskRunner(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            label_list=task.get_labels(),
            device=device,
            rparams=RunnerParameters(
                max_seq_length=args.max_seq_length,
                local_rank=args.local_rank, n_gpu=n_gpu, fp16=args.fp16,
                learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
                t_total=t_total, warmup_proportion=args.warmup_proportion,
                num_train_epochs=args.num_train_epochs,
                train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
            )
        )

    if args.do_train:
        assert at_most_one_of([args.do_val_history, args.train_save_every])
        if args.do_val_history:
            val_examples = task.get_dev_examples()
            results = runner.run_train_val(
                train_examples=train_examples,
                val_examples=val_examples,
                task_name=task.name,
            )
            metrics_str = json.dumps(results, indent=2)
            with open(os.path.join(args.output_dir, "val_metrics_history.json"), "w") as f:
                f.write(metrics_str)
        elif args.train_save_every:
            train_dataloader = runner.get_train_dataloader(train_examples, verbose=not args.not_verbose)
            for epoch in range(int(args.num_train_epochs)):
                for step, _, _ in runner.run_train_epoch_context(train_dataloader):
                    if step % args.train_save_every == args.train_save_every - 1 \
                            or step == len(train_dataloader) - 1:
                        glue_model_setup.save_bert(
                            model=model, optimizer=optimizer, args=args,
                            save_path=os.path.join(
                                args.output_dir, f"all_state___epoch{epoch:04d}___batch{step:06d}.p"
                            ),
                            save_mode=args.bert_save_mode,
                            verbose=not args.not_verbose,
                        )
        else:
            runner.run_train(train_examples)

    if args.do_save:
        # Save a trained model
        if not args.only_train_classifier:
            glue_model_setup.save_bert(
                model=model, optimizer=optimizer, args=args,
                save_path=os.path.join(args.output_dir, "all_state.p"),
                save_mode=args.bert_save_mode,
            )
        # save trained classifier and bert model
        if args.only_train_classifier:
            model_to_save = bert_as_encoder.module if hasattr(bert_as_encoder, 'module') else model
            bert_state_dict = utils.to_cpu(model_to_save.state_dict())
            torch.save(bert_state_dict, os.path.join(args.output_dir, "bert_encoder_state_dict.p"))

    if args.do_val:
        val_examples = task.get_dev_examples()
        if args.only_train_classifier:
            results = runner_classifier.run_val(val_examples, task_name=task.name, verbose=not args.not_verbose)
        else:
            results = runner.run_val(val_examples, task_name=task.name, verbose=not args.not_verbose)
        
        df = pd.DataFrame(results["logits"])
        df.to_csv(os.path.join(args.output_dir, "val_preds.csv"), header=False, index=False)
        metrics_str = json.dumps({"loss": results["loss"], "metrics": results["metrics"]}, indent=2)
        print(metrics_str)
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
            f.write(metrics_str)

        # HACK for MNLI-mismatched
        if task.name == "mnli":
            mm_val_examples = MnliMismatchedProcessor().get_dev_examples(task.data_dir)
            if args.only_train_classifier:
                mm_results = runner_classifier.run_val(mm_val_examples, task_name=task.name, verbose=not args.not_verbose)
            else:
                mm_results = runner.run_val(mm_val_examples, task_name=task.name, verbose=not args.not_verbose)
            
            df = pd.DataFrame(results["logits"])
            df.to_csv(os.path.join(args.output_dir, "mm_val_preds.csv"), header=False, index=False)
            combined_metrics = {}
            for k, v in results["metrics"].items():
                combined_metrics[k] = v
            for k, v in mm_results["metrics"].items():
                combined_metrics["mm-"+k] = v
            combined_metrics_str = json.dumps({
                "loss": results["loss"],
                "metrics": combined_metrics,
            }, indent=2)
            print(combined_metrics_str)
            with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
                f.write(combined_metrics_str)

    if args.do_test:
        test_examples = task.get_test_examples()
        logits = runner.run_test(test_examples, verbose=not args.not_verbose)
        df = pd.DataFrame(logits)
        df.to_csv(os.path.join(args.output_dir, "test_preds.csv"), header=False, index=False)

        # HACK for MNLI-mismatched
        if task.name == "mnli":
            test_examples = MnliMismatchedProcessor().get_test_examples(task.data_dir)
            logits = runner.run_test(test_examples)
            df = pd.DataFrame(logits)
            df.to_csv(os.path.join(args.output_dir, "mm_test_preds.csv"), header=False, index=False)


if __name__ == "__main__":
    main()
