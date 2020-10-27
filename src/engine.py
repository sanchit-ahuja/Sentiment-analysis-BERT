from tqdm import tqdm
import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, accumulation_steps, scheduler = None):
    model.train()

    for batch_idx,dataset in tqdm(enumerate(data_loader),total=len(data_loader)):
        ids = dataset["ids"]
        token_type_ids = dataset["token_type_ids"]
        masks = dataset["mask"]
        target = dataset["target"]

        ids = ids.to(device,dtype = torch.long)
        token_type_ids = token_type_ids.to(device,dtype = torch.long)
        mask = masks.to(device,dtype = torch.long)
        targets = targets.to(device,dtype = torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids,masks = mask,token_type_ids = token_type_ids)

        # Accumulation
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Step the optimizer after certain number of accumulation steps

        # if(batch_idx + 1)%config.ACC
        
def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad(): 
        for batch_idx,dataset in tqdm(enumerate(data_loader),total=len(data_loader)):
            ids = dataset["ids"]
            token_type_ids = dataset["token_type_ids"]
            masks = dataset["mask"]
            target = dataset["target"]

            ids = ids.to(device,dtype = torch.long)
            token_type_ids = token_type_ids.to(device,dtype = torch.long)
            mask = masks.to(device,dtype = torch.long)
            targets = targets.to(device,dtype = torch.float)

            optimizer.zero_grad()
            outputs = model(ids=ids,masks = mask,token_type_ids = token_type_ids)

            # Accumulation
            fin_