import os
import torch
from torch import nn
import torch.nn.functional as nnf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Tuple, Optional
import clip
import PIL.Image
import skimage.io as io
from PIL import Image, ImageFont, ImageDraw

device = 'cpu'
T = torch.Tensor
D = torch.device

model_path = './model_wieghts.pt'

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
    

def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def list_files(directory):
    try:
        # Get a list of all files in the directory
        files = os.listdir(directory)

        # Filter out only files (not subdirectories) and list their names with extensions
        file_list = [f for f in files if os.path.isfile(os.path.join(directory, f))]

        return file_list

    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def add_title_to_image(
    image_path,
    text,
    font_size=40,
    font_color=(255, 255, 255),  # White by default
    y_position=50,  # Distance from top
    font_path="arial.ttf"
):
    
    # Open the image
    with Image.open(image_path) as img:
        # Create a drawing object
        draw = ImageDraw.Draw(img)
        
        # Load font (use default if none specified)
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            
        # Get text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        
        # Calculate x position to center the text
        img_width = img.size[0]
        x_position = (img_width - text_width) // 2
        
        # Add text shadow for better visibility
        shadow_offset = 2
        shadow_color = (0, 0, 0)
        draw.text(
            (x_position + shadow_offset, y_position + shadow_offset), 
            text, 
            font=font, 
            fill=shadow_color
        )
        
        # Draw the main text
        draw.text(
            (x_position, y_position), 
            text, 
            font=font, 
            fill=font_color
        )
        
        return img



clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prefix_length = 10

model = ClipCaptionModel(prefix_length)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True),strict=False) 
model = model.eval() 
model = model.to(device)


# Define the path to the 'images' directory
root_directory = os.getcwd()  # Gets the current working directory
images_directory = os.path.join(root_directory, 'images')

# Call the function to list files in the 'images' directory
img_list = list_files(images_directory)

for img_filename in img_list:
    img_file = os.path.join(images_directory, img_filename)
    image = io.imread(img_file)
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    print(generated_text_prefix)
    out = add_title_to_image(img_file, generated_text_prefix,font_size=35,y_position=30)

    # Save the image in the 'output' directory
    output_directory = os.path.join(root_directory, 'output')
    out_file = os.path.join(output_directory, img_filename)
    out.save(out_file)

    # Display the image
    out.show()
