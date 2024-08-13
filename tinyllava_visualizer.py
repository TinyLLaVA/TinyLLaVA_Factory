from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel
from matplotlib import pyplot as plt
import torch
import requests
from io import BytesIO
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import numpy as np
import os
import datetime
from tinyllava.data import *
from tinyllava.utils import *
from tinyllava.model import *


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def extract_max_values_and_indices(tensor, k):
    max_values, max_indices = torch.topk(tensor, k, dim=2)
    max_values_with_indices = torch.stack((max_indices, max_values), dim=3)
    return max_values_with_indices


def visualize_grid_to_grid(i, mask, image, output_dir, grid_size=27, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    mask = mask.detach().cpu().numpy()
    mask = Image.fromarray(mask).resize((384, 384))
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(image)
    ax[0].axis('off')

    ax[1].imshow(image)
    im = ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    cbar = fig.colorbar(im, ax=ax[1])
    cbar.set_label('Color Temperature')
    name = os.path.join(output_dir, "hot_image", f"{i}.png")
    plt.savefig(name)
    plt.close(fig)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_word_images(tokenizer, top_words_tensor, num, input_ids, embed_tokens, output_dir):
    num_top_words = top_words_tensor.shape[1]
    for i in range(num_top_words - num, num_top_words):
        fig, ax = plt.subplots()
        word_indices = top_words_tensor[0, i, :, 0].detach().cpu().numpy()
        probabilities = top_words_tensor[0, i, :, 1].detach().cpu().numpy()
        colors = plt.cm.viridis(probabilities)

        for j, (word_index, color, prob) in enumerate(zip(word_indices, colors, probabilities)):
            word = tokenizer.decode([word_index])
            prob_text = f"{word}  P: {prob:.2f}"
            ax.text(0.5, 0.9 - j * 0.1, prob_text, color=color, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        ax.set_title('Top Words for Index {}'.format(i - num_top_words + num + 1))
        plt.savefig(os.path.join(output_dir, 'word', f"word_image_{i - num_top_words + num + 1}.png"))
        plt.close()


def generate_word_images_before(tokenizer, input_ids, tensor, num, top_words_tensor, output_dir):
    num_top_words = tensor.shape[2]
    result = tensor.mean(dim=1)  # [1, len, len]
    input_ids_fir = input_ids[input_ids != -200].unsqueeze(0)  # 去除了图像的token
    for i in range(num_top_words - num, num_top_words - 1):
        top1_indices = top_words_tensor[0, i, 0, 0].long()
        fig, ax = plt.subplots()
        result_1 = result[0, i, 0:input_ids.shape[1]]
        result_1 = result_1[input_ids.squeeze() != -200]
        if not i == num_top_words - num:
            result_2 = result[0, i, num_top_words - num + 1:i + 1]
            result_1 = torch.cat((result_1, result_2), dim=0)

        if not i == num_top_words - num:
            output_ids = top_words_tensor[0, num_top_words - num:i, 0, 0].unsqueeze(0).long()
            input_ids_fir = torch.cat((input_ids_fir, output_ids), dim=1)

        tv, ti = torch.topk(result_1.squeeze(), 8)
        tv = tv / torch.max(tv)
        probabilities = tv.detach().cpu().numpy()
        colors = plt.cm.viridis(probabilities)
        for j, (word_index, color, prob) in enumerate(zip(ti, colors, probabilities)):
            word = tokenizer.decode(input_ids_fir[0, word_index.item()])
            prob_text = f"{word}  P: {prob:.2f}"
            ax.text(0.5, 0.9 - j * 0.1, prob_text, color=color, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        ax.set_title(
            'similarities of output word  {}'.format(tokenizer.decode([top1_indices.detach().cpu().numpy()])))
        plt.savefig(os.path.join(output_dir, 'word_before', f"word_image_{i - (num_top_words - num - 1)}.png"))
        plt.close()


class Monitor:
    def __init__(self, args, llm_layers_index, ):
        if args.model_path is not None:
            model, tokenizer, image_processor, context_len = load_pretrained_model(args.model_path)
        else:
            assert args.model is not None, 'model_path or model must be provided'
            model = args.model
            if hasattr(model.config, "max_sequence_length"):
                context_len = model.config.max_sequence_length
            else:
                context_len = 2048
        self.model = model
        self.args = args
        self.input_ids = None
        self.image = None
        self.params = list(model.parameters())
        self.output = defaultdict(dict)
        self.attentions = []
        self.hidden = []
        self.logit = []
        self.image_token = []
        self.llm_layers_index = llm_layers_index
        self._register(llm_layers_index)

    def _register(self, llm_layers_index):
        def attention_hook(module, input, output):
            self.hidden.append(input[0])

        def output_hook(module, input, output):
            self.logit.append(output)

        def image_hook(module, input, output):
            self.image_token.append(output)

        mod = self.model
        mod.language_model.model.layers[llm_layers_index].register_forward_hook(attention_hook)
        mod.language_model.lm_head.register_forward_hook(output_hook)
        mod.connector.register_forward_hook(image_hook)

    def prepare_input(self):
        #  获得input_ids
        qs = self.args.query
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        text_processor = TextPreprocess(self.model.tokenizer, self.args.conv_mode)
        msg = Message()
        msg.add_message(qs)
        result = text_processor(msg.messages, mode='eval')
        self.input_ids = result['input_ids'].unsqueeze(0).cuda()
        #  获得图片tensor
        data_args = self.model.config
        image_processor = self.model.vision_tower._image_processor
        image_processor = ImagePreprocess(image_processor, data_args)
        image_files = self.args.image_file.split(self.args.sep)
        images = load_images(image_files)[0]
        images_tensor = image_processor(images)
        image_tensor = 255 * (images_tensor - images_tensor.min()) / (images_tensor.max() - images_tensor.min())
        image_tensor = image_tensor.clamp(0, 255)
        image_tensor = image_tensor.byte()
        to_pil = ToPILImage()
        self.image = to_pil(image_tensor).convert('RGB')
        #  预处理捕获的中间态
        self.model.cuda()
        self.logit = F.softmax(torch.cat(self.logit, dim=1), dim=2)  # 输出的词的概率
        hidden_tensor = torch.cat(self.hidden, dim=1)
        length = hidden_tensor.shape[1]
        attention_mask = torch.unsqueeze(
            torch.unsqueeze(generate_square_subsequent_mask(length).clone().detach(), dim=0),
            dim=0).cuda()

        # 获得attention map
        self.hidden = self.model.language_model.model.layers[self.llm_layers_index](hidden_tensor,
                                                                                    output_attentions=True,
                                                                                    attention_mask=attention_mask)
        self.image_token = self.image_token[0].squeeze()
        self.image_token = torch.cat((torch.zeros(1, 2560).cuda(), self.image_token), dim=0)

    def get_output(self, output_dir='results/'):
        print("Starting visualization...")
        self.prepare_input()
        # 创建唯一的时间戳目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # 创建三个子文件夹
        os.makedirs(os.path.join(output_dir, 'word'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'word_before'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'hot_image'), exist_ok=True)

        # 生成输出概率
        num = self.logit.shape[1] - 726 - len(self.input_ids[0])
        result = extract_max_values_and_indices(self.logit, 8)
        generate_word_images(self.model.tokenizer, result, num, self.input_ids,
                             self.model.language_model.model.embed_tokens.weight, output_dir)

        # llm输出和输入的词之间的关系
        generate_word_images_before(self.model.tokenizer, self.input_ids, self.hidden[1], num, result, output_dir)

        result_top1 = result[0, :, 0, 0].squeeze()
        for i in range(len(result_top1) - num, len(result_top1)):
            word_id = result_top1[i]
            word_id_tensor = torch.tensor([word_id]).long().cuda()
            word_vector = self.model.language_model.model.embed_tokens(word_id_tensor).squeeze().detach()
            vector_expanded = word_vector.unsqueeze(0).expand_as(self.image_token)
            vector_norm = F.normalize(vector_expanded, p=2, dim=1)
            matrix_norm = F.normalize(self.image_token, p=2, dim=1)
            cosine_similarities = torch.sum(vector_norm * matrix_norm, dim=1)
            normalized_similarities = F.softmax(cosine_similarities, dim=0)
            visualize_grid_to_grid('hot_image_' + str(i - (len(result_top1) - num) + 1),
                                   normalized_similarities.view(27, 27),
                                   self.image, output_dir)
        print("Completed visualization.")
