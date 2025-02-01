import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import math
import torch.onnx
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/GenClimb")
import torch.quantization

# IMPORT CONFIG, DATA, AND TOKENIZER
with open ("./configs/config_____.json", "r") as f:
    config = json.load(f)

with open('..../board_tokens_srcs.json', 'r') as f:
    src_sequences = json.load(f)

with open('..../board_tokens_tgts.json', 'r') as f:
    tgt_sequences = json.load(f)

with open("..../token_to_id.json", "r") as file:
    token_to_id = json.load(file)

VOCAB_SIZE = len(token_to_id)
with open("..../id_to_token.json", "r") as file:
    id_to_token = json.load(file)

# DATA SPLITTING
src_train_idx = int(len(src_sequences) * config['train_split'])
src_train_sequences = src_sequences[:src_train_idx]
src_eval_sequences = src_sequences[src_train_idx:]
tgt_train_sequences = tgt_sequences[:src_train_idx]
tgt_eval_sequences = tgt_sequences[src_train_idx:]

# DATASET
class ClimbsData(Dataset):
    def __init__(self, src_sequences, tgt_sequences, PAD_token, SOS_token, EOS_token):
        super().__init__()
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
        self.pred_sequences = tgt_sequences
        self.PAD_token = PAD_token
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, index):
        longest_src_sequence = max([len(seq) for seq in self.src_sequences])
        longest_tgt_sequence = max([len(seq) for seq in self.tgt_sequences]) + 1

        src = self.src_sequences[index]
        tgt = self.tgt_sequences[index]
        pred = self.pred_sequences[index]

        num_src_padding = longest_src_sequence - len(src)
        num_tgt_padding = longest_tgt_sequence - len(tgt) - 1

        pred = torch.cat([torch.tensor(pred), torch.tensor([self.EOS_token]), torch.tensor([self.PAD_token]*num_tgt_padding)]).to(dtype=torch.int64)
        tgt = torch.cat([torch.tensor([self.SOS_token]), torch.tensor(tgt), torch.tensor([self.PAD_token]*num_tgt_padding)]).to(dtype=torch.int64)
        src = torch.cat([torch.tensor(src), torch.tensor([self.PAD_token]*num_src_padding)]).to(dtype=torch.int64)

        return src, tgt, pred

# GENERATE MASKS
def get_masks(src, tgt, PAD_TOKEN):
    src_key_padding_mask = (src == PAD_TOKEN)
    tgt_key_padding_mask = (tgt == PAD_TOKEN)
    return src_key_padding_mask, tgt_key_padding_mask

# DATA
train_data = ClimbsData(src_sequences=src_train_sequences, tgt_sequences=tgt_train_sequences, PAD_token=token_to_id['<PAD>'], SOS_token=token_to_id['<SOS>'], EOS_token=token_to_id['<EOS>'])
eval_data = ClimbsData(src_sequences=src_eval_sequences, tgt_sequences=tgt_eval_sequences, PAD_token=token_to_id['<PAD>'], SOS_token=token_to_id['<SOS>'], EOS_token=token_to_id['<EOS>'])

# DATA LOADER
train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
eval_dataloader = DataLoader(eval_data, batch_size=config['batch_size'], shuffle=True)

# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout= 0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# TRANSFORMER
class Transformer(nn.Module):
    def __init__(self, dim, num_heads, num_layers, vocab_size, ff_dim, dropout, activation, layer_norm_eps, batch_first, norm_first, padding_idx, device):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.padding_idx = padding_idx
        self.device = device
        EncoderLayer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout, activation=self.activation, layer_norm_eps=self.layer_norm_eps, batch_first=self.batch_first, norm_first=self.norm_first, device=self.device)
        self.Encoder = nn.TransformerEncoder(encoder_layer=EncoderLayer, num_layers=self.num_layers)
        DecoderLayer = nn.TransformerDecoderLayer(d_model=self.dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout, activation=self.activation, layer_norm_eps=self.layer_norm_eps, batch_first=self.batch_first, norm_first=self.norm_first, device=self.device)
        self.Decoder = nn.TransformerDecoder(decoder_layer=DecoderLayer, num_layers=self.num_layers)
        self.Embedding = nn.Embedding(num_embeddings=(self.vocab_size+1), embedding_dim=self.dim, padding_idx=self.padding_idx)
        self.PositionalEncoding = PositionalEncoding(d_model=self.dim, dropout=self.dropout, max_len=1024)
        self.Linear = nn.Linear(self.dim, self.vocab_size)

    def generate_causal_mask(self, sz, dtype=torch.float32):
        mask = torch.triu(torch.full((sz, sz), float('-inf'), dtype=dtype, device=self.device), diagonal=1) == float('-inf')
        return mask
    
    def encode(self, src, src_key_padding_mask=None):
        src = self.PositionalEncoding(self.Embedding(src) * math.sqrt(self.dim))
        memory = self.Encoder(src, src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def decode(self, tgt, memory, tgt_key_padding_mask=None):
        tgt = self.PositionalEncoding(self.Embedding(tgt) * math.sqrt(self.dim))
        tgt_mask = self.generate_causal_mask(tgt.size(-2))
        x = self.Decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.Linear(x)
        return out
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        out = self.decode(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        return out

# MODEL, OPTIMIZER, LOSS_FN
model = Transformer(dim=config['dim'], num_heads=config['num_heads'], num_layers=config['num_layers'], vocab_size=VOCAB_SIZE, ff_dim=config['ff_dim'], dropout=config['dropout'], 
                    activation=config['activation'], layer_norm_eps=config['layer_norm_eps'], batch_first=config['batch_first'], norm_first=config['norm_first'], padding_idx=token_to_id['<PAD>'], device=config['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=token_to_id['<PAD>']).to(device=config['device'])

# TRAIN LOOP
def train_loop(model, dataloader, optimizer, loss_fn):
    num_batches = len(dataloader)
    
    model.train()

    for batch, (src, tgt, pred) in enumerate(tqdm(dataloader)):
        src = src.to(config['device'])
        tgt = tgt.to(config['device'])
        pred = pred.to(config['device'])

        src_key_padding_mask, tgt_key_padding_mask = get_masks(src=src, tgt=tgt, PAD_TOKEN=token_to_id['<PAD>'])

        src_key_padding_mask = src_key_padding_mask.to(config['device'])
        tgt_key_padding_mask = tgt_key_padding_mask.to(config['device'])

        logits = model(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask).to(config['device'])
        
        optimizer.zero_grad()
        loss = loss_fn(logits.view(-1, logits.size(-1)), pred.view(-1))
        writer.add_scalar('train loss', loss.item(), batch)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"\nBatch number [{batch+1}/{num_batches}]: Loss is {loss.item()}")
            writer.flush()
    return loss.item()

# EVAL LOOP
def eval_loop(model, dataloader, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (src, tgt, pred) in enumerate(tqdm(dataloader)):
            src = src.to(config['device'])
            tgt = tgt.to(config['device'])
            pred = pred.to(config['device'])

            src_key_padding_mask, tgt_key_padding_mask = get_masks(src=src, tgt=tgt, PAD_TOKEN=token_to_id['<PAD>'])

            src_key_padding_mask = src_key_padding_mask.to(config['device'])
            tgt_key_padding_mask = tgt_key_padding_mask.to(config['device'])

            logits = model(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask).to(config['device'])

            total_loss += loss_fn(logits.view(-1, logits.size(-1)), pred.view(-1)).item()
            correct += (logits.argmax(-1) == pred).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / size
    print(f"\nTest Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

# TRAINING
for e in range(config['epochs']):
    print(f"Epoch {e+1}\n-------------------------------")
    loss = train_loop(model=model, dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn)
    eval_loop(model=model, dataloader=eval_dataloader, loss_fn=loss_fn)
    filename = f"./GenClimb/GenClimb_epoch{e+1}.pt"
    torch.save({
        "epoch": e+1, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filename)
filename = "./GenClimb/GenClimb.pt"
torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
}, filename)
writer.close()

############################## QUANTIZE #################################################
checkpoint = torch.load("./GenClimb/GenClimb.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={nn.Linear, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer},  # Only quantize Linear layers
    dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "./GenClimb/GenClimb-Quantized.bin")

############################## ONNX #################################################
tgt_seq_len = 1
src = torch.randint(0, VOCAB_SIZE, (1, 2)).to(config['device']).to(dtype=torch.int64)
tgt = torch.randint(0, VOCAB_SIZE, (1, tgt_seq_len)).to(config['device']).to(dtype=torch.int64)

torch.onnx.export(model=model, args=(src, tgt), f="./GenClimb/GenClimb.onnx", export_params=True, do_constant_folding=True, input_names=['src', 'tgt'], output_names=['output'], opset_version=14, dynamic_axes={
    "tgt" : {1: 'tgt_seq_len'}
})