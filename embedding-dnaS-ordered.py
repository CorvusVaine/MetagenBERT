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

def extract(file):
    f=open(file,"r")
    L_files_not_ok = []
    L_files_ok = []
    lines = f.readlines()
    for l in lines:
        if "not" in l:
            L_files_not_ok.append(l.split(" ")[0])
        else:
            L_files_ok.append(l.split(" ")[0])
    print(L_files_ok)
    print(L_files_not_ok)
    return L_files_not_ok

def extract_to_embed(file):
    f=open(file,"r")
    L_to_embed = f.readlines()
    L_to_embed = [l.strip("\n") for l in L_to_embed ]
    return L_to_embed

def extract_list(l):
    return l[1:-1].split(",")

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
    
def main(tokenizer_path,config_path,model_path,sequence_dir,saving_path,to_avoid,batch_size=10000):
    #Create all paths
    print("mkdir")
    if idr_torch.rank == 0 and idr_torch.local_rank==0:
        for f in os.listdir(sequence_dir):
            saving_file_path = saving_path+f[:-6]
            if not os.path.exists(saving_file_path):
                os.makedirs(saving_file_path)
            mean_saving_path = os.path.join(saving_file_path, 'mean/')
#            max_saving_path = os.path.join(saving_file_path, 'max/')
            idx_saving_path = os.path.join(saving_file_path, 'idx/')
            if not os.path.exists(mean_saving_path):
                os.makedirs(mean_saving_path)
#            if not os.path.exists(max_saving_path):
#                os.makedirs(max_saving_path)
            if not os.path.exists(idx_saving_path):
                os.makedirs(idx_saving_path)
    print("parall")
    
    dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size=idr_torch.size, 
                        rank=idr_torch.rank)
#    D_time={}
    print("rank :")
    print(idr_torch.rank)
    print("local rank :")
    print(idr_torch.local_rank)
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    ## Load model and tokenizer
    mean_saving_path = os.path.join(saving_path, 'mean')
#    max_saving_path = os.path.join(saving_path, 'max')
    idx_saving_path = os.path.join(saving_path, 'idx')
#    max_length = 60
    tdeb = time.time()
#    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, max_length=max_length, padding = "max_length", truncation = True)
    config = BertConfig.from_pretrained(config_path,revision='main')
#    max_length = config.max_position_embeddings
    max_length = 42
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, max_length=max_length, padding = "max_length", truncation = True)
#    config.max_position_embeddings = max_length+2
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,revision='main',attn_implementation = 'eager')#, config=config)
# Replacing hub model by downloaded version to leave flah attention out, keep only if sure
    #model = AutoModel.from_pretrained("/lustre/fswork/projects/rech/lyt/uce73iu/DNABERT_2_revision/DNABERT-2-117M", trust_remote_code=True)#, attn_implementation = 'eager' )#,config=config_dna2)

    model = model.to(gpu)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DDP(model, device_ids=[idr_torch.local_rank])
    model.eval()
    L_files = os.listdir(sequence_dir)
    L_files.sort()
#    L_files = ["cleaned_ERR527190_1.fasta","cleaned_ERR527144_1.fasta","cleaned_ERR527177_1.fasta", "cleaned_ERR527178_1.fasta", "cleaned_ERR527176_1.fasta","cleaned_ERR528299_1.fasta", "cleaned_ERR528300_1.fasta","cleaned_ERR528310_1.fasta","cleaned_ERR528295_1.fasta","cleaned_ERR527035_1.fasta","cleaned_ERR527012_1.fasta", "cleaned_ERR527011_1.fasta","cleaned_ERR527034_1.fasta","cleaned_ERR527062_1.fasta","cleaned_ERR527082_1.fasta","cleaned_ERR527139_1.fasta","cleaned_ERR527190_2.fasta","cleaned_ERR527144_2.fasta","cleaned_ERR527177_2.fasta", "cleaned_ERR527178_2.fasta", "cleaned_ERR527176_2.fasta","cleaned_ERR528299_2.fasta", "cleaned_ERR528300_2.fasta","cleaned_ERR528310_2.fasta","cleaned_ERR528295_2.fasta","cleaned_ERR527035_2.fasta","cleaned_ERR527012_2.fasta", "cleaned_ERR527011_2.fasta","cleaned_ERR527034_2.fasta","cleaned_ERR527062_2.fasta","cleaned_ERR527082_2.fasta","cleaned_ERR527139_2.fasta"]
# Uncomment for avoiding certain files
    #L_to_embed = extract_to_embed(to_avoid)
    L_to_embed = L_files
    #L_to_embed = extract_list(to_avoid)
    print(L_to_embed)
    flag = open(saving_path+"/flag.txt", 'a')
    for sequence_file in L_files:
        t_samp=time.time()
        batch_index=0
        print("rank",idr_torch.rank,sequence_file)
        saving_file_path = saving_path + sequence_file[:-6]
        sequence_file = sequence_dir+sequence_file
        mean_saving_path = os.path.join(saving_file_path, 'mean/')
#        max_saving_path = os.path.join(saving_file_path, 'max/')
        idx_saving_path = os.path.join(saving_file_path, 'idx/')
        if sequence_file.split("/")[-1] not in L_to_embed:
            print(sequence_file,"considered already treated")
            continue
        print(saving_file_path.split("/")[-1])
        #if saving_file_path.split("/")[-1] in L_to_embed:
        print("Go")
        dataset = SentenceDataset(sequence_file, tokenizer, max_length)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length",max_length=max_length)
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                num_replicas=idr_torch.size,
                                                                rank=idr_torch.rank,
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
 #           if sequence_file.split("/")[-1] in D_time.keys():
  #              D_time[sequence_file.split("/")[-1]]=D_time[sequence_file.split("/")[-1]].append(time.time()-t_samp)
   #         else: 
    #            D_time[sequence_file.split("/")[-1]]=time.time()-t_samp
     #       print(time.time()-tdeb)
    #if idr_torch.rank == 0:
     #   D_all_times = [None] * idr_torch.size
      #  dist.gather_object(D_time, D_all_times, dst=0)
       # D_final={}
        #for dict in D_all_times:
         #   for fi in dict.keys():
          #      if fi not in D_final.keys():
           #         D_final[fi]=[dict[fi]]
            #    else:
             #       D_final[fi].append(dict[fi])
        #D_true_final={}
        #for ech in D_final.keys():
         #   print(D_final[ech])
          #  D_true_final[ech]=sum(D_final[ech])/len(D_final[ech])
        #print("Collected Batch Times: ", D_final)
        #print("Collected Batch Times: ", D_true_final)
    #else:
     #   dist.gather_object(D_time, dst=0)
        #else: 
         #   print("already treated")
#    all_embeddings = torch.load(os.path.join(mean_saving_path, f'embeddings_{0}.pt'))
#    os.remove(os.path.join(mean_saving_path, f'embeddings_{0}.pt'))
#    for i in range(1,batch_index):
#        all_embeddings = torch.cat((all_embeddings,torch.load(os.path.join(mean_saving_path, f'embeddings_{i}.pt'))),dim=0)
#        os.remove(os.path.join(mean_saving_path, f'embeddings_{i}.pt'))
#    torch.save(all_embeddings, os.path.join(mean_saving_path, 'embeddings.pt'))
#    all_embeddings = torch.load(os.path.join(max_saving_path, f'embeddings_{0}.pt'))
#    os.remove(os.path.join(max_saving_path, f'embeddings_{0}.pt'))
#    for i in range(1,batch_index):
#        all_embeddings = torch.cat((all_embeddings,torch.load(os.path.join(max_saving_path, f'embeddings_{i}.pt'))),dim=0)
#        os.remove(os.path.join(max_saving_path, f'embeddings_{i}.pt'))
#    torch.save(all_embeddings, os.path.join(max_saving_path, 'embeddings.pt'))

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
    args = parser.parse_args()
    main(args.tokenizer_path,args.config_path,args.model_path,args.sequence_dir,args.saving_path,args.to_avoid,args.batch_size)
