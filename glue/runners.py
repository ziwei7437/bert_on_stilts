import collections as col
import logging
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as pl

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .core import InputFeatures, InputFeaturesSeparated, Batch, BatchSeparated, InputExample, TokenizedExample
from .evaluate import compute_metrics
from pytorch_pretrained_bert.utils import truncate_seq_pair
from shared.runners import warmup_linear

logger = logging.getLogger(__name__)


class LabelModes:
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"


class TrainEpochState:
    def __init__(self):
        self.tr_loss = 0
        self.global_step = 0
        self.nb_tr_examples = 0
        self.nb_tr_steps = 0


class TrainingState:
    def __init__(self):
        self.tr_loss = list()
        self.val_history = list()
        self.epoch_loss = list()

    def draw_loss_curve(self):
        """plot loss in every mini-batch"""
        pl.figure()
        pl.plot(self.tr_loss)
        pl.grid()

    def draw_val_history(self):
        """plot validation accuracy for each epoch"""
        pl.figure()
        pl.plot(self.val_history)
        pl.grid()

def tokenize_example(example, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
    else:
        tokens_b = example.text_b
    return TokenizedExample(
        guid=example.guid,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        label=example.label,
    )


def convert_example_to_feature_separated(example, tokenizer, max_seq_length, label_map):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a, tokens_b = example.tokens_a, example.tokens_b
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]
    if len(tokens_b) > max_seq_length - 2:
        tokens_b = tokens_b[:(max_seq_length - 2)]

    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
    tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
    segment_ids_a = [0] * len(tokens_a)
    segment_ids_b = [0] * len(tokens_b)

    input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
    input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

    input_mask_a = [1] * len(input_ids_a)
    input_mask_b = [1] * len(input_ids_b)

    padding_a = [0] * (max_seq_length - len(input_ids_a))
    input_ids_a += padding_a
    input_mask_a += padding_a
    segment_ids_a += padding_a

    padding_b = [0] * (max_seq_length - len(input_ids_b))
    input_ids_b += padding_b
    input_mask_b += padding_b
    segment_ids_b += padding_b

    assert len(input_ids_a) == max_seq_length
    assert len(input_mask_a) == max_seq_length
    assert len(segment_ids_a) == max_seq_length
    assert len(input_ids_b) == max_seq_length
    assert len(input_mask_b) == max_seq_length
    assert len(segment_ids_b) == max_seq_length

    if is_null_label_map(label_map):
        label_id = example.label
    else:
        label_id = label_map[example.label]

    return InputFeaturesSeparated(
        guid=example.guid,
        input_ids_a=input_ids_a,
        input_ids_b=input_ids_b,
        input_mask_a=input_mask_a,
        input_mask_b=input_mask_b,
        segment_ids_a=segment_ids_a,
        segment_ids_b=segment_ids_b,
        label_id=label_id,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
    )


def convert_example_to_feature(example, tokenizer, max_seq_length, label_map):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a, tokens_b = example.tokens_a, example.tokens_b
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if is_null_label_map(label_map):
        label_id = example.label
    else:
        label_id = label_map[example.label]
    return InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        tokens=tokens,
    )


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, verbose=True):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_feature(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in feature_instance.tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in feature_instance.input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in feature_instance.input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in feature_instance.segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, feature_instance.label_id))

        features.append(feature_instance)
    return features


def convert_examples_to_features_separated(examples, label_map, max_seq_length, tokenizer, verbose=True):
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_feature_separated(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens_a: %s" % " ".join([str(x) for x in feature_instance.tokens_a]))
            logger.info("input_ids_a: %s" % " ".join([str(x) for x in feature_instance.input_ids_a]))
            logger.info("input_mask_a: %s" % " ".join([str(x) for x in feature_instance.input_mask_a]))
            logger.info(
                "segment_ids_a: %s" % " ".join([str(x) for x in feature_instance.segment_ids_a]))

            logger.info("tokens_b: %s" % " ".join([str(x) for x in feature_instance.tokens_b]))
            logger.info("input_ids_b: %s" % " ".join([str(x) for x in feature_instance.input_ids_b]))
            logger.info("input_mask_b: %s" % " ".join([str(x) for x in feature_instance.input_mask_b]))
            logger.info(
                "segment_ids_b: %s" % " ".join([str(x) for x in feature_instance.segment_ids_b]))

            logger.info("label: %s (id = %d)" % (example.label, feature_instance.label_id))
        features.append(feature_instance)
    return features


def convert_to_dataset(features, label_mode):
    full_batch = features_to_data(features, label_mode=label_mode)
    if full_batch.label_ids is None:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids)
    else:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids, full_batch.label_ids)
    return dataset, full_batch.tokens


def convert_to_dataset_separated(features, label_mode):
    full_batch = features_to_data_separated(features, label_mode=label_mode)
    if full_batch.label_ids is None:
        dataset = TensorDataset(full_batch.input_ids_a, full_batch.input_mask_a,
                                full_batch.segment_ids_a, full_batch.input_ids_b,
                                full_batch.input_mask_b, full_batch.segment_ids_b)
    else:
        dataset = TensorDataset(full_batch.input_ids_a, full_batch.input_mask_a,
                                full_batch.segment_ids_a,
                                full_batch.input_ids_b, full_batch.input_mask_b,
                                full_batch.segment_ids_b,
                                full_batch.label_ids)
    return dataset, full_batch.tokens_a, full_batch.tokens_b


def features_to_data(features, label_mode):
    if label_mode == LabelModes.CLASSIFICATION:
        label_type = torch.long
    elif label_mode == LabelModes.REGRESSION:
        label_type = torch.float
    else:
        raise KeyError(label_mode)
    return Batch(
        input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long),
        input_mask=torch.tensor([f.input_mask for f in features], dtype=torch.long),
        segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        label_ids=torch.tensor([f.label_id for f in features], dtype=label_type),
        tokens=[f.tokens for f in features],
    )


def features_to_data_separated(features, label_mode):
    if label_mode == LabelModes.CLASSIFICATION:
        label_type = torch.long
    elif label_mode == LabelModes.REGRESSION:
        label_type = torch.float
    else:
        raise KeyError(label_mode)
    return BatchSeparated(
        input_ids_a=torch.tensor([f.input_ids_a for f in features], dtype=torch.long),
        input_mask_a=torch.tensor([f.input_mask_a for f in features], dtype=torch.long),
        segment_ids_a=torch.tensor([f.segment_ids_a for f in features], dtype=torch.long),
        label_ids=torch.tensor([f.label_id for f in features], dtype=label_type),
        tokens_a=[f.tokens_a for f in features],
        tokens_b=[f.tokens_b for f in features],
        input_ids_b=torch.tensor([f.input_ids_b for f in features], dtype=torch.long),
        input_mask_b=torch.tensor([f.input_mask_b for f in features], dtype=torch.long),
        segment_ids_b=torch.tensor([f.segment_ids_b for f in features], dtype=torch.long),
    )


class HybridLoader:
    def __init__(self, dataloader, tokens):
        self.dataloader = dataloader
        self.tokens = tokens

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 4:
                input_ids, input_mask, segment_ids, label_ids = batch
            elif len(batch) == 3:
                input_ids, input_mask, segment_ids = batch
                label_ids = None
            else:
                raise RuntimeError()
            batch_tokens = self.tokens[i * batch_size: (i + 1) * batch_size]
            yield Batch(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                tokens=batch_tokens,
            )

    def __len__(self):
        return len(self.dataloader)


class HybridLoaderSeparated:
    def __init__(self, dataloader, tokens_a, tokens_b):
        self.dataloader = dataloader
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 7:
                input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b, label_ids = batch
            elif len(batch) == 6:
                input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b = batch
                label_ids = None
            else:
                raise RuntimeError()
            batch_tokens_a = self.tokens_a[i * batch_size: (i + 1) * batch_size]
            batch_tokens_b = self.tokens_b[i * batch_size: (i + 1) * batch_size]
            yield BatchSeparated(
                input_ids_a=input_ids_a,
                input_mask_a=input_mask_a,
                segment_ids_a=segment_ids_a,
                label_ids=label_ids,
                tokens_a=batch_tokens_a,
                tokens_b=batch_tokens_b,
                input_ids_b=input_ids_b,
                input_mask_b=input_mask_b,
                segment_ids_b=segment_ids_b,
            )

    def __len__(self):
        return len(self.dataloader)


class RunnerParameters:
    def __init__(self, max_seq_length, local_rank, n_gpu, fp16,
                 learning_rate, gradient_accumulation_steps, t_total, warmup_proportion,
                 num_train_epochs, train_batch_size, eval_batch_size):
        self.max_seq_length = max_seq_length
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.t_total = t_total
        self.warmup_proportion = warmup_proportion
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size


class GlueTaskRunner:
    def __init__(self, model, optimizer, tokenizer, label_list, device, rparams):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_map = {v: i for i, v in enumerate(label_list)}
        self.device = device
        self.rparams = rparams

    def run_train(self, train_examples, verbose=True):
        if verbose:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)
            logger.info("  Num steps = %d", self.rparams.t_total)
        train_dataloader = self.get_train_dataloader(train_examples, verbose=verbose)

        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader)

    def run_train_val(self, train_examples, val_examples, task_name):
        epoch_result_dict = col.OrderedDict()
        for i in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            train_dataloader = self.get_train_dataloader(train_examples, verbose=False)
            self.run_train_epoch(train_dataloader)
            epoch_result = self.run_val(val_examples, task_name, verbose=False)
            del epoch_result["logits"]
            epoch_result_dict[i] = epoch_result
        return epoch_result_dict

    def run_train_epoch(self, train_dataloader):
        for _ in self.run_train_epoch_context(train_dataloader):
            pass

    def run_train_epoch_context(self, train_dataloader):
        self.model.train()
        train_epoch_state = TrainEpochState()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state):
        batch = batch.to(self.device)
        loss = self.model(batch.input_ids, batch.segment_ids, batch.input_mask, batch.label_ids)
        if self.rparams.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.rparams.gradient_accumulation_steps > 1:
            loss = loss / self.rparams.gradient_accumulation_steps
        if self.rparams.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += batch.input_ids.size(0)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.rparams.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = self.rparams.learning_rate * warmup_linear(
                train_epoch_state.global_step / self.rparams.t_total, self.rparams.warmup_proportion)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_epoch_state.global_step += 1

    def run_val(self, val_examples, task_name, verbose=True):
        val_dataloader = self.get_eval_dataloader(val_examples, verbose=verbose)
        self.model.eval()
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        all_labels = []
        for step, batch in enumerate(tqdm(val_dataloader, desc="Evaluating (Val)")):
            batch = batch.to(self.device)

            with torch.no_grad():
                tmp_eval_loss = self.model(batch.input_ids, batch.segment_ids,
                                           batch.input_mask, batch.label_ids)
                logits = self.model(batch.input_ids, batch.segment_ids, batch.input_mask)
                label_ids = batch.label_ids.cpu().numpy()

            logits = logits.detach().cpu().numpy()
            total_eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += batch.input_ids.size(0)
            nb_eval_steps += 1
            all_logits.append(logits)
            all_labels.append(label_ids)
        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return {
            "logits": all_logits,
            "loss": eval_loss,
            "metrics": compute_task_metrics(task_name, all_logits, all_labels),
        }

    def run_test(self, test_examples, verbose=True):
        test_dataloader = self.get_eval_dataloader(test_examples, verbose=verbose)
        self.model.eval()
        all_logits = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="Predictions (Test)")):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = self.model(batch.input_ids, batch.segment_ids, batch.input_mask)
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits

    def get_train_dataloader(self, train_examples, verbose=True):
        train_features = convert_examples_to_features(
            train_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        train_data, train_tokens = convert_to_dataset(
            train_features, label_mode=get_label_mode(self.label_map),
        )
        if self.rparams.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.rparams.train_batch_size,
        )
        return HybridLoader(train_dataloader, train_tokens)

    def get_eval_dataloader(self, eval_examples, verbose=True):
        eval_features = convert_examples_to_features(
            eval_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        eval_data, eval_tokens = convert_to_dataset(
            eval_features, label_mode=get_label_mode(self.label_map),
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoader(eval_dataloader, eval_tokens)


def compute_task_metrics(task_name, logits, labels):
    if logits.shape[1] == 1:
        pred_arr = logits.reshape(-1)
    else:
        pred_arr = np.argmax(logits, axis=1)
    return compute_metrics(
        task_name=task_name,
        pred_srs=pred_arr,
        label_srs=labels,
    )


def is_null_label_map(label_map):
    return len(label_map) == 1 and label_map[None] == 0


def get_label_mode(label_map):
    if is_null_label_map(label_map):
        return LabelModes.REGRESSION
    else:
        return LabelModes.CLASSIFICATION


class GlueTaskClassifierRunner:
    def __init__(self, bert_model, classifier_model, optimizer, tokenizer,
                 label_list, device, rparams, train_infer_classifier):
        self.bert_model = bert_model
        self.classifier_model = classifier_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_map = {v: i for i, v in enumerate(label_list)}
        self.device = device
        self.rparams = rparams
        self.train_infer_classifier = train_infer_classifier
        self.training_state = TrainingState()

    def run_train_classifier(self, train_examples, verbose=True):
        if verbose:
            logger.info("***** Running Training for Classifier *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)
            logger.info("  Num steps = %d", self.rparams.t_total)

        if self.train_infer_classifier:
            train_dataloader = self.get_train_dataloader_separated(train_examples, verbose=verbose)
        else:
            train_dataloader = self.get_train_dataloader(train_examples, verbose=verbose)

        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader)

    def run_train_epoch(self, train_dataloader):
        for step, batch, train_epoch_state in self.run_train_epoch_context(train_dataloader):
            pass
        self.training_state.epoch_loss.append(train_epoch_state.tr_loss)

    def run_train_epoch_context(self, train_dataloader):
        self.classifier_model.train()
        self.bert_model.eval()
        train_epoch_state = TrainEpochState()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state):

        batch = batch.to(self.device)
        self.bert_model.eval()
        self.classifier_model.train()

        if not self.train_infer_classifier:
            with torch.no_grad():
                _, pooled_output = self.bert_model(batch.input_ids, batch.segment_ids, batch.input_mask,
                                                   output_all_encoded_layers=False)
            loss = self.classifier_model(pooled_output, batch.label_ids)
        else:
            with torch.no_grad():
                _, pooled_output_a = self.bert_model(batch.input_ids_a, batch.segment_ids_a, batch.input_mask_a,
                                                     output_all_encoded_layers=False)
                _, pooled_output_b = self.bert_model(batch.input_ids_b, batch.segment_ids_b, batch.input_mask_b,
                                                     output_all_encoded_layers=False)
            loss = self.classifier_model(pooled_output_a, pooled_output_b, batch.label_ids)

        if self.rparams.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.rparams.gradient_accumulation_steps > 1:
            loss = loss / self.rparams.gradient_accumulation_steps
        if self.rparams.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        self.training_state.tr_loss.append(loss.item())
        print("Mini-batch Loss: {:.4f}".format(self.training_state.tr_loss[-1]))

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += batch.input_ids.size(0) if not self.train_infer_classifier else batch.input_ids_a.size(0)
        train_epoch_state.nb_tr_steps += 1

        if (step + 1) % self.rparams.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = self.rparams.learning_rate * warmup_linear(
                train_epoch_state.global_step / self.rparams.t_total, self.rparams.warmup_proportion)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_epoch_state.global_step += 1

    def run_val(self, val_examples, task_name, verbose=True):

        if self.train_infer_classifier:
            val_dataloader = self.get_eval_dataloader_separated(val_examples, verbose=verbose)
        else:
            val_dataloader = self.get_eval_dataloader(val_examples, verbose=verbose)

        self.classifier_model.eval()
        self.bert_model.eval()

        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        all_labels = []
        for step, batch in enumerate(tqdm(val_dataloader, desc="Evaluating (Val)")):
            batch = batch.to(self.device)

            if self.train_infer_classifier:
                with torch.no_grad():
                    _, pooled_output_a = self.bert_model(batch.input_ids_a, batch.segment_ids_a, batch.input_mask_a,
                                                         output_all_encoded_layers=False)
                    _, pooled_output_b = self.bert_model(batch.input_ids_b, batch.segment_ids_b, batch.input_mask_b,
                                                         output_all_encoded_layers=False)
                    tmp_eval_loss = self.classifier_model(pooled_output_a, pooled_output_b, batch.label_ids)
                    logits = self.classifier_model(pooled_output_a, pooled_output_b)
                    label_ids = batch.label_ids.cpu().numpy()

            else:
                with torch.no_grad():
                    _, pooled_output = self.bert_model(batch.input_ids, batch.segment_ids, batch.input_mask,
                                                       output_all_encoded_layers=False)
                    tmp_eval_loss = self.classifier_model(pooled_output, batch.label_ids)
                    logits = self.classifier_model(pooled_output)
                    label_ids = batch.label_ids.cpu().numpy()

            logits = logits.detach().cpu().numpy()
            total_eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += batch.input_ids.size(0) if not self.train_infer_classifier else batch.input_ids_a.size(0)
            nb_eval_steps += 1
            all_logits.append(logits)
            all_labels.append(label_ids)

        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return {
            "logits": all_logits,
            "loss": eval_loss,
            "metrics": compute_task_metrics(task_name, all_logits, all_labels),
        }

    def get_train_dataloader(self, train_examples, verbose=True):
        train_features = convert_examples_to_features(
            train_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        train_data, train_tokens = convert_to_dataset(
            train_features, label_mode=get_label_mode(self.label_map),
        )
        if self.rparams.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.rparams.train_batch_size,
        )
        return HybridLoader(train_dataloader, train_tokens)

    def get_train_dataloader_separated(self, train_examples, verbose=True):
        train_features = convert_examples_to_features_separated(
            train_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose
        )
        train_data, train_tokens_a, train_tokens_b = convert_to_dataset_separated(
            train_features, label_mode=get_label_mode(self.label_map)
        )
        if self.rparams.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.rparams.train_batch_size,
        )
        return HybridLoaderSeparated(train_dataloader, train_tokens_a, train_tokens_b)

    def get_eval_dataloader(self, eval_examples, verbose=True):
        eval_features = convert_examples_to_features(
            eval_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        eval_data, eval_tokens = convert_to_dataset(
            eval_features, label_mode=get_label_mode(self.label_map),
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoader(eval_dataloader, eval_tokens)

    def get_eval_dataloader_separated(self, eval_examples, verbose=True):
        eval_features = convert_examples_to_features_separated(
            eval_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        eval_data, eval_tokens_a, eval_tokens_b = convert_to_dataset_separated(
            eval_features, label_mode=get_label_mode(self.label_map),
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoaderSeparated(eval_dataloader, eval_tokens_a, eval_tokens_b)
