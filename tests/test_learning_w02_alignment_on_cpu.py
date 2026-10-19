"""学习批次切分与合并；这里只操作本地数据，没有启动 worker。"""

from collections import Counter

import numpy as np
import pytest
import torch

from tests.learning_batches import make_batch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def test_chunk_concat_preserves_samples():
    data = make_batch(16)

    # chunk(4) 的 4 是份数：把 16 条样本分成 4 份，每份 4 条。
    parts = data.chunk(4)
    assert len(parts) == 4
    assert [len(part) for part in parts] == [4, 4, 4, 4]

    for part_index, part in enumerate(parts):
        start = part_index * 4
        stop = start + 4
        # 逐份验证身份与顺序，而不只是检查每份都有 4 条。
        assert part.non_tensor_batch["sample_id"].tolist() == [f"s{i:03d}" for i in range(start, stop)]
        for key in data.batch.keys():
            assert torch.equal(part.batch[key], data.batch[key][start:stop])
        for key in data.non_tensor_batch:
            assert np.array_equal(part.non_tensor_batch[key], data.non_tensor_batch[key][start:stop])
        assert part.meta_info == data.meta_info

    # concat 按列表顺序拼接，不会根据 sample_id 自动排序。
    restored = DataProto.concat(parts)

    assert len(restored) == 16
    assert set(restored.batch.keys()) == set(data.batch.keys())
    assert set(restored.non_tensor_batch) == set(data.non_tensor_batch)
    for key in data.batch.keys():
        assert restored.batch[key].dtype == data.batch[key].dtype
        assert torch.equal(restored.batch[key], data.batch[key])
    for key in data.non_tensor_batch:
        assert restored.non_tensor_batch[key].dtype == data.non_tensor_batch[key].dtype
        assert np.array_equal(restored.non_tensor_batch[key], data.non_tensor_batch[key])
    assert restored.meta_info == data.meta_info


def test_chunk_rejects_uneven_batch_without_padding():
    data = make_batch(17)
    # 当前未启用自动 padding；chunk 要求等分，17 不能整除 4。
    assert not data.is_padding_enabled()
    with pytest.raises(AssertionError, match="only support equal chunk"):
        data.chunk(4)


def test_restore_order_after_out_of_order_collection():
    """按完成顺序收集后，利用样本 ID 恢复分发前的顺序。"""
    original = make_batch(16)
    original_ids = original.non_tensor_batch["sample_id"].tolist()
    parts = original.chunk(4)

    # 仅模拟收集顺序：第 2 块先返回，然后是第 0、3、1 块。
    completion_order = [2, 0, 3, 1]
    collected = DataProto.concat([parts[i] for i in completion_order])
    collected_ids = collected.non_tensor_batch["sample_id"].tolist()

    # shape、样本身份和出现次数都正确，但顺序确实已经改变。
    assert collected.batch["input_ids"].shape == original.batch["input_ids"].shape
    assert Counter(collected_ids) == Counter(original_ids)
    assert collected_ids != original_ids

    # 建立 “ID -> 收集后的行位置” 映射。
    # 这种映射要求 ID 唯一：如果有重复，字典只会保留最后一个位置。
    assert len(set(original_ids)) == len(original_ids)
    position_by_id = {sample_id: position for position, sample_id in enumerate(collected_ids)}

    # 按分发前保存的 ID 顺序查询位置，不依赖 ID 的字符串排序规则。
    restore_indices = torch.tensor([position_by_id[sample_id] for sample_id in original_ids], dtype=torch.long)
    assert restore_indices.tolist() == [4, 5, 6, 7, 12, 13, 14, 15, 0, 1, 2, 3, 8, 9, 10, 11]

    # 整个 DataProto 一起重排，不能只修正 sample_id。
    collected.reorder(restore_indices)

    for key in original.batch.keys():
        assert torch.equal(collected.batch[key], original.batch[key])
    for key in original.non_tensor_batch:
        assert np.array_equal(collected.non_tensor_batch[key], original.non_tensor_batch[key])
    assert collected.meta_info == original.meta_info


@pytest.mark.parametrize(
    ("interleave", "expected_rows"),
    [
        pytest.param(True, [0, 0, 1, 1, 2, 2, 3, 3], id="repeat-each-row"),
        pytest.param(False, [0, 1, 2, 3, 0, 1, 2, 3], id="repeat-whole-batch"),
    ],
)
def test_repeat_preserves_alignment_and_expected_counts(interleave, expected_rows):
    """合法重复需要检查次数、顺序和字段对应，不能要求 ID 唯一。"""
    original = make_batch(4)
    original_ids = original.non_tensor_batch["sample_id"].tolist()

    # repeat 返回新 DataProto；它只重复已有数据，不调用模型生成回答。
    repeated = original.repeat(repeat_times=2, interleave=interleave)

    assert repeated is not original
    assert len(original) == 4
    assert original.non_tensor_batch["sample_id"].tolist() == original_ids
    assert len(repeated) == 8

    repeated_ids = repeated.non_tensor_batch["sample_id"].tolist()
    assert Counter(repeated_ids) == {sample_id: 2 for sample_id in original_ids}
    assert repeated_ids == [f"s{i:03d}" for i in expected_rows]

    # 用明确的行索引表达期望结果，验证所有字段按相同规则重复。
    expected_indices = torch.tensor(expected_rows, dtype=torch.long)
    for key in original.batch.keys():
        assert torch.equal(repeated.batch[key], original.batch[key][expected_indices])
    for key in original.non_tensor_batch:
        assert np.array_equal(repeated.non_tensor_batch[key], original.non_tensor_batch[key][expected_rows])
    assert repeated.meta_info == original.meta_info


@pytest.mark.parametrize(("n", "expected_pad_size"), [(16, 0), (17, 3)])
def test_pad_chunk_concat_unpad_preserves_original(n, expected_pad_size):
    """补齐样本数用于等分，按原块顺序收集合并后删除补齐行。"""
    original = make_batch(n)

    # 这是补齐样本行数，不是给每条序列补 token。
    padded, pad_size = pad_dataproto_to_divisor(original, size_divisor=4)
    assert pad_size == expected_pad_size
    assert len(padded) == n + expected_pad_size
    assert len(padded) % 4 == 0

    # 当前实现把开头的样本复制到末尾；17 行时末尾追加第 0、1、2 行。
    expected_rows = list(range(n)) + list(range(expected_pad_size))
    expected_indices = torch.tensor(expected_rows, dtype=torch.long)
    assert padded.non_tensor_batch["sample_id"].tolist() == [f"s{i:03d}" for i in expected_rows]
    for key in original.batch.keys():
        assert torch.equal(padded.batch[key], original.batch[key][expected_indices])
    for key in original.non_tensor_batch:
        assert np.array_equal(padded.non_tensor_batch[key], original.non_tensor_batch[key][expected_rows])

    parts = padded.chunk(4)
    assert len(parts) == 4
    assert [len(part) for part in parts] == [len(padded) // 4] * 4

    # 本例不改变样本内容；保持分发顺序合并，使补齐行仍位于末尾。
    collected = DataProto.concat(parts)
    restored = unpad_dataproto(collected, pad_size)

    # unpad 不识别重复 ID，只按 pad_size 去掉尾部行。
    # pad_size=0 时应保留全部数据，不能直接使用 data[:-0]（会得到空批次）。
    assert len(restored) == n
    for key in original.batch.keys():
        assert torch.equal(restored.batch[key], original.batch[key])
    for key in original.non_tensor_batch:
        assert np.array_equal(restored.non_tensor_batch[key], original.non_tensor_batch[key])
    assert restored.meta_info == original.meta_info
