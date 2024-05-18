class BaseTrainingRecipe:
    """
    Load and save model, set some parameters in the model.
    """
    def call(self, model):
        """
        Set the parameters required for training.

        Parameters
        ----------
        model :
            the models include llm, Connector, VisionTower
        """

    def load(self, model, model_args):
        """
        Load llm, visiontower, connector and pretrained model.

        Parameters
        ----------
        model:
            the models include llm, Connector, VisionTower
        model_args : dict
            model arguments
        """

    def save(self, model, trainer):
        """
        Save llm, visiontower, connector, tokenizer, trainer and entire model config.

        Parameters
        ----------
        model :
            the models include llm, Connector, VisionTower
        trainer :
            trainer for the model
        """

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into chunks of roughly equal lengths.

    **Returns**: a list containing several chunks of roughly equal lengths

    **Return type**: list

    Parameters
    ----------
    indices : list
        a list contains index
    lengths : list
        a list of lengths corresponding to indices
    num_chunks : int
        the num of chunks to be split
    """

def get_length_grouped_indices(lengths, batch_size, world_size, generator):
    """
    Generate batches of index lists according to lengths.

    **Returns**: a list containing batches of indices

    **Return type**: list

    Parameters
    ----------
    lengths : list
        a list of lengths corresponding to indices
    batch_size : int
        the size of each batch
    world_size : int
    generator :
        generator for generating random numbers
    """

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator):
    """
    Call get_length_grouped_indices to generate large batches of index lists containing both multimodal and language samples according to lengths.

    **Returns**: a list containing large batches of indices

    **Return type**: list

    Parameters
    ----------
    lengths : list
        a list of lengths corresponding to indices
    batch_size : int
        the size of each batch
    world_size : int
    generator :
        generator for generating random numbers
    """

class LengthGroupedSampler:
    """
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while keeping a bit of randomness.

    **Returns**: an iterator for iterating over the sampled indices

    **Return type**: iterator

    Parameters
    ----------
    batch_size : int
        the size of each batch
    world_size : int
    lengths : list
        a list of lengths corresponding to indices
    generator :
        generator for generating random numbers
    group_by_modality : bool
        bool for determining whether to group by modality
    """

class LLaVATrainer:
    """
    Call LengthGroupedSampler to sample train dataset, and setup the optimizer.

    **Returns**: trainer

    Parameters
    ----------
    model
    tokenizer : transformers.PreTrainedTokenizer
        tokenize data
    training_arguments : dict
        training arguments
    """