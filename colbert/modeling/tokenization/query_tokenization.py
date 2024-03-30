import torch

from colbert.modeling.hf_colbert import class_factory
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.utils import _split_into_batches
from colbert.utils.utils import batch
from colbert.parameters import DEVICE


class QueryTokenizer:
    def __init__(self, config: ColBERTConfig, verbose: int = 3, instruction_model=None):
        HF_ColBERT = class_factory(config.checkpoint)
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)
        self.verbose = verbose

        self.config = config
        self.query_maxlen = config.query_maxlen
        self.background_maxlen = (
            512 - self.query_maxlen + 1
        )  # FIXME: Make this configurable

        self.Q_marker_token, self.Q_marker_token_id = (
            config.query_token,
            self.tok.convert_tokens_to_ids(config.query_token_id),
        )
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = (
            self.tok.mask_token,
            self.tok.mask_token_id,
        )
        self.pad_token, self.pad_token_id = self.tok.pad_token, self.tok.pad_token_id
        self.used = False

        self.instruction_tokenizer = (
            HF_ColBERT.instruction_tokenizer_from_pretrained(instruction_model)
            if instruction_model
            else None
        )

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [
            prefix
            + lst
            + suffix
            + [self.mask_token] * (self.query_maxlen - (len(lst) + 3))
            for lst in tokens
        ]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        ids = self.tok(batch_text, add_special_tokens=False).to(DEVICE)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = (
            [self.cls_token_id, self.Q_marker_token_id],
            [self.sep_token_id],
        )
        ids = [
            prefix
            + lst
            + suffix
            + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3))
            for lst in ids
        ]

        return ids

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instructions: {task_description}\nQuery: {query}'

    def tensorize(
        self,
        batch_text,
        bsize=None,
        context=None,
        full_length_search=False,
        has_instructions: bool = False,
        instructions_separator: str = "[__ENDINSTR__]",
    ):
        assert type(batch_text) in [list, tuple], type(batch_text)

        instructions_batch_text = []
        # print('BATCH TEXT')
        # print(batch_text)
        # print(batch_text)
        if has_instructions:
            for i, text in enumerate(batch_text):
                before, after = text.split(instructions_separator)
                if '[SEP]' in before:
                    before = self.get_detailed_instruct(before.split('[SEP]')[0], before.split('[SEP]')[1])
                batch_text[i] = after.lstrip()
                instructions_batch_text.append(before.rstrip())

        # print(len(batch_text))
        # print(batch_text)
        # print(instructions_batch_text)
        # print(len(instructions_batch_text))
        # assert len(batch_text) == len(instructions_batch_text)

        # add placehold for the [Q] marker
        batch_text = [". " + x for x in batch_text]

        # Full length search is only available for single inference (for now)
        # Batched full length search requires far deeper changes to the code base
        assert full_length_search == False or (
            type(batch_text) == list and len(batch_text) == 1
        )

        if full_length_search:
            # Tokenize each string in the batch
            un_truncated_ids = self.tok(batch_text, add_special_tokens=False).to(
                DEVICE
            )["input_ids"]
            # Get the longest length in the batch
            max_length_in_batch = max(len(x) for x in un_truncated_ids)
            # Set the max length
            max_length = self.max_len(max_length_in_batch)
        else:
            # Max length is the default max length from the config
            max_length = self.query_maxlen

        obj = self.tok(
            batch_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(DEVICE)

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == self.pad_token_id] = self.mask_token_id

        if context is not None:
            assert len(context) == len(batch_text), (len(context), len(batch_text))

            obj_2 = self.tok(
                context,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=self.background_maxlen,
            ).to(DEVICE)

            ids_2, mask_2 = (
                obj_2["input_ids"][:, 1:],
                obj_2["attention_mask"][:, 1:],
            )  # Skip the first [SEP]

            ids = torch.cat((ids, ids_2), dim=-1)
            mask = torch.cat((mask, mask_2), dim=-1)

        if self.config.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        if has_instructions:
            # print(instructions_batch_text)  
            # print(type(self.instruction_tokenizer))
            # print(instructions_batch_text)

            # NO QWEN
            # instruction_encodings = self.instruction_tokenizer(
            #     instructions_batch_text,
            #     padding=True,
            #     truncation=True,
            #     return_tensors="pt",
            #     max_length=512,
            # ).to(DEVICE)

            # QWEN
            batch_dict = self.instruction_tokenizer(instructions_batch_text, max_length=510, return_attention_mask=False, padding=False, truncation=True)
            # append eos_token_id to every input_ids
            # Original line: batch_dict['input_ids'] = [input_ids + [self.instruction_tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
            repeat_indices = [len(input_ids) for input_ids in batch_dict['input_ids']]
            batch_dict['input_ids'] = [input_ids + [self.instruction_tokenizer.eos_token_id] + input_ids + [self.instruction_tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
            batch_dict = self.instruction_tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
            instruction_ids = batch_dict['input_ids']
            # instruction_ids = instruction_encodings['input_ids']
            # print('instruction ids earlier')
            # print(instruction_ids)
            instruction_masks =  batch_dict['attention_mask']
            # instruction_masks = instruction_encodings['attention_mask']

        if bsize:
            if has_instructions:
                batches = _split_into_batches(ids, mask, bsize, instruction_ids, instruction_masks)
            else:
                batches = _split_into_batches(ids, mask, bsize)
            return batches

        if self.used is False:
            self.used = True

            firstbg = (context is None) or context[0]
            if self.verbose > 1:
                print()
                print(
                    "#> QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) =="
                )
                print(f"#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}")
                print(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
                print(f"#> Output Mask: {mask[0].size()}, {mask[0]}")
                print()

        if has_instructions:
            # print('query ids')
            # print(ids)
            # print('instruction ids')
            # print(instruction_ids)
            return ids, mask, instruction_ids, instruction_masks
        return ids, mask

    # Ensure that query_maxlen <= length <= 500 tokens
    def max_len(self, length):
        return min(500, max(self.query_maxlen, length))
