"""
Donut model wrapper for document information extraction.

Donut (Document Understanding Transformer) is an OCR-free model that uses a
Swin Transformer encoder + BART decoder to go directly from document images
to structured JSON output. No OCR step needed.

Paper: https://arxiv.org/abs/2111.15664
Pretrained: naver-clova-ix/donut-base

Usage:
    from src.models.donut import DonutModelWrapper

    model = DonutModelWrapper(
        pretrained_name="naver-clova-ix/donut-base",
        max_length=768,
        image_size=(1280, 960),
        dataset_name="cord",
    )

    # Training
    loss = model.training_step(pixel_values, labels, decoder_input_ids)

    # Inference
    json_output = model.generate(pixel_values, task_prompt="<s_cord-v2>")
"""

import json
import re
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)


# Task prompt tokens for each dataset -- Donut uses these to tell the decoder
# what kind of document it's looking at and what schema to produce.
TASK_PROMPTS = {
    "cord": "<s_cord-v2>",
    "funsd": "<s_funsd>",
    "sroie": "<s_sroie>",
}

# Special tokens added to the decoder vocabulary for each dataset.
# These correspond to the JSON keys in the ground truth annotations.
DATASET_SPECIAL_TOKENS = {
    "cord": [
        "<s_cord-v2>", "</s_cord-v2>",
        "<s_menu>", "</s_menu>",
        "<s_sub_total>", "</s_sub_total>",
        "<s_total>", "</s_total>",
        "<s_unitprice>", "</s_unitprice>",
        "<s_cnt>", "</s_cnt>",
        "<s_discountprice>", "</s_discountprice>",
        "<s_price>", "</s_price>",
        "<s_itemsubtotal>", "</s_itemsubtotal>",
        "<s_tax_price>", "</s_tax_price>",
        "<s_nm>", "</s_nm>",
        "<s_num>", "</s_num>",
        "<s_vatyn>", "</s_vatyn>",
        "<s_etc>", "</s_etc>",
        "<s_menutype_cnt>", "</s_menutype_cnt>",
        "<s_menuqty_cnt>", "</s_menuqty_cnt>",
        "<s_sub_etc>", "</s_sub_etc>",
        "<s_service_price>", "</s_service_price>",
        "<s_emoney_price>", "</s_emoney_price>",
        "<s_creditcardprice>", "</s_creditcardprice>",
        "<s_changeprice>", "</s_changeprice>",
        "<s_menuetc_price>", "</s_menuetc_price>",
        "<s_void_menu>", "</s_void_menu>",
    ],
    "funsd": [
        "<s_funsd>", "</s_funsd>",
        "<s_question>", "</s_question>",
        "<s_answer>", "</s_answer>",
        "<s_header>", "</s_header>",
        "<s_other>", "</s_other>",
    ],
    "sroie": [
        "<s_sroie>", "</s_sroie>",
        "<s_company>", "</s_company>",
        "<s_date>", "</s_date>",
        "<s_address>", "</s_address>",
        "<s_total>", "</s_total>",
    ],
}


class DonutModelWrapper(nn.Module):
    """
    Wrapper around HuggingFace's VisionEncoderDecoderModel for Donut.

    Handles:
    - Loading pretrained weights and processor
    - Adding dataset-specific special tokens to the decoder vocab
    - Resizing encoder input resolution
    - Training step (forward + loss)
    - Generation (image -> JSON string)
    - JSON parsing from the generated token sequence
    """

    def __init__(
            self,
            pretrained_name: str = "naver-clova-ix/donut-base",
            max_length: int = 768,
            image_size: tuple[int, int] = (1280, 960),
            dataset_name: str = "cord",
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.max_length = max_length
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.task_prompt = TASK_PROMPTS[dataset_name]

        # Load processor (handles image preprocessing + tokenization)
        self.processor = DonutProcessor.from_pretrained(pretrained_name)
        self.processor.image_processor.size = {
            "height": image_size[0],
            "width": image_size[1],
        }
        self.processor.image_processor.do_align_long_axis = False

        # Load the VisionEncoderDecoder model
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_name)

        # Add dataset-specific special tokens to decoder tokenizer
        special_tokens = DATASET_SPECIAL_TOKENS.get(dataset_name, [])
        if special_tokens:
            newly_added = self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )
            if newly_added > 0:
                self.model.decoder.resize_token_embeddings(
                    len(self.processor.tokenizer)
                )

        # Update model config
        self.model.config.encoder.image_size = list(image_size)
        self.model.config.decoder_start_token_id = (
            self.processor.tokenizer.convert_tokens_to_ids([self.task_prompt])[0]
        )
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(
            self,
            pixel_values: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass. Returns model outputs (includes loss if labels given)."""
        return self.model(
            pixel_values=pixel_values,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )

    def training_step(
            self,
            pixel_values: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single training step. Returns the loss tensor.

        Args:
            pixel_values: [B, 3, H, W] preprocessed images
            labels: [B, seq_len] token IDs for ground truth JSON sequence.
                    Padding positions should be set to -100.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.Tensor,
            num_beams: int = 1,
            max_length: Optional[int] = None,
    ) -> list[str]:
        """
        Generate structured JSON strings from document images.

        Args:
            pixel_values: [B, 3, H, W] preprocessed images
            num_beams: beam search width (1 = greedy)
            max_length: override max generation length

        Returns:
            List of JSON strings, one per image in the batch.
        """
        max_length = max_length or self.max_length

        decoder_input_ids = torch.full(
            (pixel_values.size(0), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=pixel_values.device,
        )

        outputs = self.model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )

        # Decode token IDs to strings
        decoded = self.processor.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )

        # Parse each decoded string into clean JSON
        results = []
        for seq in decoded:
            parsed = self.token_sequence_to_json(seq)
            results.append(parsed)

        return results

    def token_sequence_to_json(self, token_sequence: str) -> str:
        """
        Convert Donut's generated token sequence into a JSON string.

        Donut outputs something like:
            <s_cord-v2><s_menu><s_nm>AMERICANO</s_nm><s_price>3.00</s_price>
            </s_menu><s_total><s_price>3.00</s_price></s_total></s_cord-v2>

        This converts it to:
            {"menu": [{"nm": "AMERICANO", "price": "3.00"}],
             "total": {"price": "3.00"}}
        """
        # Remove the EOS and task tokens
        token_sequence = token_sequence.replace(
            self.processor.tokenizer.eos_token, ""
        ).strip()

        # Remove the outer task prompt wrapper
        task_end = f"</{self.task_prompt[1:]}"  # e.g. </s_cord-v2>
        token_sequence = token_sequence.replace(self.task_prompt, "")
        token_sequence = token_sequence.replace(task_end, "")

        return self._parse_tokens_recursive(token_sequence)

    def _parse_tokens_recursive(self, token_str: str) -> str:
        """
        Recursively parse Donut's XML-like token sequence into JSON.

        Handles nested structures and repeated keys (converted to lists).
        """
        result = {}
        # Match opening/closing tag pairs
        pattern = r"<s_(\w+)>(.*?)</s_\1>"

        while True:
            match = re.search(pattern, token_str, re.DOTALL)
            if not match:
                break

            key = match.group(1)
            value_str = match.group(2).strip()

            # Check if the value contains nested tags
            if re.search(r"<s_\w+>", value_str):
                parsed_value = json.loads(
                    self._parse_tokens_recursive(value_str)
                )
            else:
                parsed_value = value_str

            # Handle repeated keys by converting to list
            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(parsed_value)
            else:
                result[key] = parsed_value

            # Remove the matched portion and continue
            token_str = token_str[: match.start()] + token_str[match.end() :]

        return json.dumps(result)

    def prepare_input(self, image, ground_truth_json: Optional[str] = None):
        """
        Prepare a single example for training or inference.

        Args:
            image: PIL Image
            ground_truth_json: JSON string for training labels (None for inference)

        Returns:
            dict with pixel_values and optionally labels
        """
        # Process the image
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        result = {"pixel_values": pixel_values}

        if ground_truth_json is not None:
            # Convert JSON to token sequence for the decoder target
            token_sequence = self.json_to_token_sequence(ground_truth_json)
            # Add task prompt at the beginning
            full_sequence = self.task_prompt + token_sequence

            # Tokenize
            tokenized = self.processor.tokenizer(
                full_sequence,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            labels = tokenized.input_ids.squeeze(0)
            # Replace padding token IDs with -100 so they're ignored in loss
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            result["labels"] = labels

        return result

    def json_to_token_sequence(self, json_str: str) -> str:
        """
        Convert a JSON string to Donut's XML-like token sequence.

        This is the inverse of token_sequence_to_json. Used to create
        training labels from ground truth annotations.

        Example:
            {"nm": "AMERICANO", "price": "3.00"}
            -> <s_nm>AMERICANO</s_nm><s_price>3.00</s_price>
        """
        if isinstance(json_str, str):
            try:
                obj = json.loads(json_str)
            except json.JSONDecodeError:
                return json_str
        else:
            obj = json_str

        return self._dict_to_tokens(obj)

    def _dict_to_tokens(self, obj) -> str:
        """Recursively convert a dict/list/str to Donut token sequence."""
        if isinstance(obj, dict):
            tokens = ""
            for key, value in obj.items():
                tokens += f"<s_{key}>"
                tokens += self._dict_to_tokens(value)
                tokens += f"</s_{key}>"
            return tokens
        elif isinstance(obj, list):
            # Each list item gets the same wrapping from the parent call
            return "".join(self._dict_to_tokens(item) for item in obj)
        else:
            return str(obj)

    def save_pretrained(self, save_dir: str):
        """Save model and processor for later loading."""
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, save_dir: str, dataset_name: str = "cord", **kwargs):
        """Load a previously saved fine-tuned model."""
        processor = DonutProcessor.from_pretrained(save_dir)
        image_size = (
            processor.image_processor.size.get("height", 1280),
            processor.image_processor.size.get("width", 960),
        )
        wrapper = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.processor = processor
        wrapper.model = VisionEncoderDecoderModel.from_pretrained(save_dir)
        wrapper.dataset_name = dataset_name
        wrapper.task_prompt = TASK_PROMPTS[dataset_name]
        wrapper.max_length = kwargs.get("max_length", 768)
        wrapper.image_size = image_size
        wrapper.pretrained_name = save_dir
        return wrapper