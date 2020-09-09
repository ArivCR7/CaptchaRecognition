import config
from tqdm import tqdm
import torch

def train(model, dataloader, optimizer):
    model.train()
    fin_loss = 0
    tk0 = tqdm(dataloader, total=len(dataloader))
    for data in tk0:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
        #eval(model, dataloader)
    return fin_loss / len(dataloader)

def eval(model, dataloader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(dataloader, total=len(dataloader))
    with torch.no_grad():
        for data in tk0:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)    
            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(dataloader)



