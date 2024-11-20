import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Model
import clip
from openl3 import load_audio_embedding  # Assuming CLAP embeddings
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from PIL import Image
import os

# Parameters
EMBEDDING_DIM = 512  # Embedding dimension for CLIP/CLAP
GPT2_DIM = 768       # GPT-2 embedding dimension
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 5e-4
COCO_ROOT = 'path_to_coco_dataset'  # Set the path to MS COCO
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# GPT-2 tokenizer and model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2").to(DEVICE)

# Dataset class
class COCODataset:
    def __init__(self, root, split='train'):
        self.coco = COCO(os.path.join(root, f'annotations/captions_{split}2017.json'))
        self.root = os.path.join(root, f'{split}2017')
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = preprocess

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_info['file_name'])
        
        # Load and preprocess image
        image = self.transform(Image.open(path).convert('RGB'))
        
        # Get caption
        captions = self.coco.imgToAnns[img_id]
        caption = captions[0]['caption']
        
        # CLIP image embedding
        with torch.no_grad():
            clip_embedding = clip_model.encode_image(image.unsqueeze(0).to(DEVICE))
        
        # GPT-2 text embedding
        tokens = gpt2_tokenizer(caption, return_tensors='pt', truncation=True, max_length=20)
        with torch.no_grad():
            gpt2_embedding = gpt2_model(**tokens.to(DEVICE)).last_hidden_state.mean(dim=1)
        
        return clip_embedding.squeeze(0), gpt2_embedding.squeeze(0)

# Load data
train_dataset = COCODataset(COCO_ROOT, split='train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define the mapping model
class MappingTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):
        super(MappingTransformer, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=4,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_in(x))
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x, x)
        x = self.fc_out(x.squeeze(1))
        return x

# Initialize model, optimizer, and loss
model = MappingTransformer(EMBEDDING_DIM, GPT2_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        clip_emb, gpt2_emb = batch
        clip_emb, gpt2_emb = clip_emb.to(DEVICE), gpt2_emb.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(clip_emb)
        loss = criterion(outputs, gpt2_emb)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} - Training Loss: {epoch_loss:.4f}")
