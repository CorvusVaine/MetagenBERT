import numpy as np
import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers.models.bert.configuration_bert import BertConfig
import time
import os
from torch.utils.data import Dataset, DataLoader
import gc
import idr_torch 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

class SentenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.sentences = self._load_sentences(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_sentences(self, file_path):
        with open(file_path, 'r') as f:
            sentences = [line.strip() for line in f]
        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded_input = self.tokenizer(sentence, padding="max_length", truncation=True, max_length=self.max_length)
        encoded_input['idx'] = idx
        return encoded_input
    
def embed(rank, tokenizer_path,config_path,model_path,sequence_dir,saving_path,batch_size=10000, world_size=1):
    #Create all saving paths
    if rank == 0:
        for f in os.listdir(sequence_dir):
            saving_file_path = saving_path+f[:-6]
            if not os.path.exists(saving_file_path):
                os.makedirs(saving_file_path)
            mean_saving_path = os.path.join(saving_file_path, 'mean/')
            idx_saving_path = os.path.join(saving_file_path, 'idx/')
            if not os.path.exists(mean_saving_path):
                os.makedirs(mean_saving_path)
            #if not os.path.exists(max_saving_path):
            #    os.makedirs(max_saving_path)
            if not os.path.exists(idx_saving_path):
                os.makedirs(idx_saving_path)    
    dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size= world_size, 
                        rank=rank)
    torch.cuda.set_device(rank)
    gpu = torch.device("cuda")
    ## Load model and tokenizer
    mean_saving_path = os.path.join(saving_path, 'mean')
    idx_saving_path = os.path.join(saving_path, 'idx')
    config = BertConfig.from_pretrained(config_path,revision='main')
    max_length = 42
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, max_length=max_length, padding = "max_length", truncation = True)
#    config.max_position_embeddings = max_length+2
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,revision='main',attn_implementation = 'eager')#, config=config)
    model = model.to(gpu)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DDP(model, device_ids=[idr_torch.local_rank])
    model.eval()
    L_files = os.listdir(sequence_dir)
    L_files.sort()
    for sequence_file in L_files:
        batch_index=0
        saving_file_path = saving_path + sequence_file[:-6]
        sequence_file = sequence_dir+sequence_file
        mean_saving_path = os.path.join(saving_file_path, 'mean/')
#        max_saving_path = os.path.join(saving_file_path, 'max/')
        idx_saving_path = os.path.join(saving_file_path, 'idx/')
        dataset = SentenceDataset(sequence_file, tokenizer, max_length)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length",max_length=max_length)
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                num_replicas=world_size,
                                                                rank=rank,
                                                                shuffle=False)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=data_sampler,
                                            collate_fn=data_collator)
        with torch.no_grad():  # Disable gradient calculation for inference
            for batch in data_loader:
                # Tokenize the batch
                gpu_batch = {k: v.to(gpu) for k, v in batch.items()}
                #gpu_batch = {k: torch.from_numpy(np.asarray(v)).to(gpu) for k, v in batch.items()}
                #gpu_batch = batch.to(gpu)
                batch_embeddings = model(**gpu_batch)[0]
                #batch_embeddings = model(**gpu_batch).last_hidden_states
                torch.save(torch.mean(batch_embeddings,dim=1).cpu(), os.path.join(mean_saving_path, f'embeddings_{batch_index}_{idr_torch.rank}.pt'))
#                torch.save(torch.max(batch_embeddings,dim=1).values.cpu(), os.path.join(max_saving_path, f'embeddings_{batch_index}_{idr_torch.rank}.pt'))
                torch.save(gpu_batch['idx'],os.path.join(idx_saving_path,f'idx_{batch_index}_{idr_torch.rank}.pt'))
                del gpu_batch
                del batch_embeddings
                torch.cuda.empty_cache()
                batch_index+=1
                gc.collect()

def main(args):
    world_size = args.world_size
    mp.spawn(embed, args=(args.tokenizer_path,args.config_path,args.model_path,args.sequence_dir,args.saving_path,args.batch_size,args.world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cleaned_Embed sequences using a pretrained model.")
    # Add an argument for the directory path
    parser.add_argument("tokenizer_path", type=str, help="Path to the tokenizer")
    parser.add_argument("config_path", type=str, help="Path to config")
    parser.add_argument("model_path", type=str, help="Path to model")
    parser.add_argument("sequence_dir", type=str, help="Sequences directory of files to embed file")
    parser.add_argument("saving_path", type=str, help="Where to save the embedded sequences")
    parser.add_argument("to_avoid", type=str, help="File to extract which to avoid")
    parser.add_argument("batch_size", type=int, default=10000, help="Batch size for embedding")
    parser.add_argument("world_size", type=int, help="Number of processes")
    args = parser.parse_args()
    main(args)


