# -*- coding: utf-8 -*-
class Template:
    """
    Get the messages, and call this function to load them in the format required by the model.

    **Returns**: tokenized prompt or label

    **Return type**: dict

    Parameters
    ----------
    messages : dict
        the conversations between question and answer
    tokenizer : transformers.PreTrainedTokenizer
        tokenize prompt
    mode : str
        train or eval
    """


class TextPreprocess:
    """
    Call Template to load and process text data.
    """

    def init(self, tokenizer, version):
        """
        Initialize tokenizer and template.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            Tokenize prompt
        version :
            The version of the template
        """

    def call(self, messages, mode='train'):
        """
        Call template.encode to process text data.

        **Returns** : tokenized prompt or label

        **Return type** : dict

        Parameters
        ----------
        messages : dict
            the conversations between human and gpt
        mode : str
            train or eval
        """


class ImagePreprocess:
    """
    To preprocess images and adjust them to different aspect ratios and resolutions.
    """

    def init(self, image_processor, data_args):
        """
        Initialize image processor, image aspect ratios and image resolutions.

        Parameters
        ----------
        image_processor :
            the processor to process image
        data_args : dcit
            data arguments
        """

    def call(self, image):
        """
        Preprocess images according to data arguments.

        **Returns** : a tensor containing the processed image patches

        **Return type** : tensor

        Parameters
        ----------
        image : PIL.Image
            the input image to be processed
        """


class LazySupervisedDataset:
    """
    Dataset for supervised fine-tuning.
    """

    def init(self, data_path, tokenizer, data_args):
        """
        Initialize tokenizer and the preprocessor of text and image.

        Parameters
        ----------
        data_path : str
            path to data
        tokenizer : transformers.PreTrainedTokenizer
            tokenize data
        data_args : dict
            data arguments
        """


class DataCollatorForSupervisedDataset:

    def call(self, instances):
        """
        Collate examples for supervised fine-tuning.

        **Returns**: a batch containing input_ids, label, attention_mask

        **Return type**: dict

        Parameters
        ----------
        instances : list(dict)
            a list of instance
        """


def make_supervised_data_module(tokenizer, data_args):
    """
    Make dataset and collator for supervised fine-tuning.

    **Returns**: a dict containing train_dataset and data_collator

    **Return type**: dict

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        tokenize data
    data_args : dict
        data arguments
    """


