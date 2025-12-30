import os
import pickle
import re
from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np


def generate_embeddings(data_path):
    # 加载测试集.pkl 文件 PP-1001_Train
    # with open("/home/aita8180/data/mtl/AGAT-PPIS/nFeature/PP-250_Test.pkl", 'rb') as f:
    #     train_data = pickle.load(f)

    with open("/home/aita8180/data/mtl/AGAT-PPIS/nFeature/PP-1001_Train.pkl", 'rb') as f:
        train_data = pickle.load(f)

    # 创建保存嵌入的目录
    os.makedirs(data_path, exist_ok=True)


    # 初始化 tokenizer 和 model（确保使用 PyTorch 权重）
    tokenizer = T5Tokenizer.from_pretrained(
        "/home/aita8180/data/mntdata/yinqy/PGAT-ABPp-main/train/Rostlab/prot_t5_xl_uniref50"
    )
    model = T5EncoderModel.from_pretrained(
        "/home/aita8180/data/mntdata/yinqy/PGAT-ABPp-main/train/Rostlab/prot_t5_xl_uniref50"
    )
    model.eval()  # 设置为推理模式

    # 如果有 GPU 可用，使用 GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for ID, data in train_data.items():
        seq = data[0]

        # 将蛋白质序列处理成用空格分隔的字母形式
        seq_processed = ' '.join(list(seq))
        seq_processed = re.sub(r"[UZOB]", "X", seq_processed)

        # 分词并转换为 PyTorch 张量
        inputs = tokenizer(
            seq_processed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 模型前向传播获取 embedding
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)

        # 将结果转为 CPU numpy 数组
        embedding = embedding.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()

        # 根据 attention mask 去除 padding 位置
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emb = embedding[seq_num][:seq_len - 1]  # 去掉最后一个 <eos> token
            features.append(seq_emb)

        features = np.concatenate(features, axis=0)

        # 保存到 .npy 文件
        np.save(os.path.join(data_path, f"{ID}.npy"), features)
        print(f"{ID} - successful")

    print("All embeddings generated.")


def main():
    data_path = "./Embeddingsss"
    generate_embeddings(data_path)


if __name__ == "__main__":
    main()
