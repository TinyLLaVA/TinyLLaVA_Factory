def register_connector(name):
    """
    Register the connector model to ConnectorFactory.

    Parameters
    ----------
    name : str
        name of connector
    """


def register_llm(name):
    """
    Register the llm model to LLMFactory.

    Parameters
    ----------
    name : str
        name of llm
    """


def register_vision_tower(name):
    """
    Register the VisionTower model to VisionTowerFactory.

    Parameters
    ----------
    name : str
        name of VisionTower
    """


def VisionTowerFactory(vision_tower_name):
    """
    Get VisionTower model according to vision_tower_name.

    **Returns**: model

    Parameters
    ----------
    vision_tower_name : str
        name of VisionTower
    """


def LLMFactory(model_name_or_path):
    """
    Get llm model according to model_name_or_path.

    **Returns**: model

    Parameters
    ----------
    model_name_or_path : str
        name or path of llm
    """


def ConnectorFactory(connector_name):
    """
    Get connector model according to connector_name

    **Returns**: model

    Parameters
    ----------
    connector_name : str
        name of connector
    """


class VisionTower:
    """
    Load the VisionTower model by vision_tower_name, extract the image feature.
    """

    def init(self, cfg):
        """
        Initialize VisionTower model.

        Parameters
        ----------
        cfg : dict
            config
        """

    def load_model(self, vision_tower_name):
        """
        Load VisionTower model.

        Parameters
        ----------
        vision_tower_name : str
            name of model
        """

    def forward(self, x):
        """
        Extract the image feature.

        **Returns**: image features

        **Return type**: tensor

        Parameters
        ----------
        x : tensor
            input image
        """


class Connector:
    """
    Load the Connector model and weights.
    """

    def init(self, config):
        """
        Initialize connector model.

        Parameters
        ----------
        config : dict
        """


class TinyLlavaPreTrainedModel:
    """
    Create pretrained TinyLLava model using TinyLlavaConfig configuration.

    Parameters
    ----------
    TinyLlavaConfig : dict
        the configuration of TinyLLava model
    """


class TinyLlavaForConditionalGeneration:
    """
    Inherit from TinyLlavaPreTrainedModel class, create pretrained TinyLLava model.
    """

    def init(self, TinyLlavaConfig):
        """
        Initialize tokenizer, llm, Connector, VisionTower.

        Parameters
        ----------
        TinyLlavaConfig : dict
            the configuration of TinyLLava model
        """

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache,
                output_attentions, output_hidden_states, images, image_sizes, return_dict):
        """
        Call language_model.forward to get the probability of the next token in the vocabulary.

        **Returns**: the probability of the next token in the vocabulary

        **Return type**: Tuple or CausalLMOutputWithPast

        Parameters
        ----------
        input_ids : tensor
            tensor containing input token id
        attention_mask : tensor
            tensor containing attention mask
        position_ids : tensor
            tensor containing position id
        past_key_values : list(tensor)
            list of key-value pairs from the previous time step
        inputs_embeds : tensor
            tensor containing input embeddings
        labels : tensor
            tensor containing labels
        use_cache : bool
            bool for determining whether to use cache
        output_attentions : bool
            bool for determining whether to output attention
        output_hidden_states : bool
            bool for determining whether to hidden states
        images : tensor
            input image
        image_sizes : list
            the size of input image
        return_dict : bool
            bool for determining whether to return in the form of dict
        """

    def generate(self, inputs, images, image_sizes):
        """
        Call language_model.generate to generate answer

        **Returns**: answer

        **Return type**: Tuple or CausalLMOutputWithPast

        Parameters
        ----------
        inputs : tensor
            input token id
        images : tensor
            input image
        image_sizes : tensor
            the size of input image
        """

    def encode_images(self, images):
        """
        Extract the image feature.

        **Returns**: image features

        **Return type**: tensor

        Parameters
        ----------
        images : tensor
            input image
        """

    def prepare_inputs_for_generation(self, input_ids, past_key_values, inputs_embeds):
        """
        Prepare input token id and input image for generation.

        **Returns**: a dict containing input token id and input image

        **Return type**: dict

        Parameters
        ----------
        images : tensor
            input image
        """

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels,
                                             images, image_sizes):
        """
        Prepare inputs containing text and image data for multimodal processing.

        Returns
        ----------
        position_ids : tensor
            the position id after processing
        attention_mask : tensor
            the attention mask after processing
        past_key_values : list(tensor)
            list of key-value pairs same as input
        new_input_embeds : tensor
            the input embeddings after processing
        new_labels : tensor
            the label after processing

        Parameters
        ----------
        input_ids : tensor
            tensor containing input token id
        position_ids : tensor
            tensor containing position id
        attention_mask : tensor
            tensor containing attention mask
        past_key_values : list(tensor)
            list of key-value pairs from the previous time step
        labels : tensor
            tensor containing labels
        images : tensor
            input image
        image_sizes : list
            the size of input image
        """