"""学习测试共用的合成样本；每次调用都创建独立数据。"""

import torch

from verl import DataProto


def make_batch(n: int = 16) -> DataProto:
    # 每行 8 个整数，模拟 token ID。数据是确定的，不需要随机种子。
    input_ids = torch.arange(n * 8, dtype=torch.long).reshape(n, 8)
    # 第一列专门记录行号，用于检查张量和 sample_id 的身份对应关系。
    input_ids[:, 0] = torch.arange(n, dtype=torch.long)

    # 本例每个位置都有效，没有 padding，所以 mask 全为 1。
    attention_mask = torch.ones_like(input_ids)

    return DataProto.from_dict(
        tensors={"input_ids": input_ids, "attention_mask": attention_mask},
        non_tensors={
            "sample_id": [f"s{i:03d}" for i in range(n)],
            # 每 4 条样本为一组；这里仅模拟分组关系，不代表真实生成结果。
            "prompt_id": [f"p{i // 4:03d}" for i in range(n)],
        },
        meta_info={"run_id": "learning-w01", "schema_version": 1},
    )
