import pytest
import torch

from tests.learning_batches import make_batch
from verl import DataProto


def test_reorder_rows():
    x = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]
    )
    order = torch.tensor([2, 0, 3, 1])

    actual = x[order]

    expected = torch.tensor(
        [
            [4, 5],
            [0, 1],
            [6, 7],
            [2, 3],
        ]
    )

    assert torch.equal(actual, expected)

    inverse_order = torch.argsort(order)
    restored = actual[inverse_order]

    assert torch.equal(restored, x)


def test_dataproto_reorder_keeps_sample_alignment():
    """重排一批样本时，张量和样本 ID 必须一起移动。"""
    # 1. 准备：每行是一条样本。第一列特意放行号，方便追踪身份。
    # 这里的 input_ids 是教学用合成数据，不是真实分词结果。
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor(
                [
                    [0, 10],  # A
                    [1, 11],  # B
                    [2, 12],  # C
                    [3, 13],  # D
                ]
            ),
        },
        non_tensors={"sample_id": ["A", "B", "C", "D"]},
        meta_info={"split": "train", "step": 7},
    )

    # tensors 存入 batch（TensorDict）；non_tensors 存入 non_tensor_batch。
    # from_dict 会把这里的 sample_id 列表转换成 NumPy 数组。
    assert len(data) == 4
    assert data.batch["input_ids"].shape == (4, 2)
    assert data.non_tensor_batch["sample_id"].tolist() == ["A", "B", "C", "D"]

    # 2. 操作：reorder 直接修改 data，不返回一个新的 DataProto。
    # 因此不要写成 data = data.reorder(order)。
    order = torch.tensor([2, 0, 3, 1])
    data.reorder(order)

    # 3. 验证：张量与 ID 都按同一个排列变化。
    expected_ids = torch.tensor(
        [
            [2, 12],  # C
            [0, 10],  # A
            [3, 13],  # D
            [1, 11],  # B
        ]
    )
    assert torch.equal(data.batch["input_ids"], expected_ids)
    assert data.non_tensor_batch["sample_id"].tolist() == ["C", "A", "D", "B"]

    # 额外检查业务对应关系：每行的编号必须匹配它的 sample_id。
    # 即使 shape 和长度都正确，单独重排一个字段仍会破坏这个关系。
    row_number_by_id = {"A": 0, "B": 1, "C": 2, "D": 3}
    for row, sample_id in zip(data.batch["input_ids"], data.non_tensor_batch["sample_id"], strict=True):
        assert row[0].item() == row_number_by_id[sample_id], f"样本 {sample_id} 与张量行错配"

    # meta_info 描述整批数据，不参与按样本重排。
    assert data.meta_info == {"split": "train", "step": 7}


def test_dataproto_rejects_mismatched_sample_count():
    """4 行张量只配 3 个样本 ID，构造时就应该被拒绝。"""
    input_ids = torch.tensor(
        [
            [0, 10],
            [1, 11],
            [2, 12],
            [3, 13],
        ]
    )

    # pytest.raises 表示这里预期出现异常；不报错反而会让测试失败。
    # match 同时检查错误信息，确认异常确实来自样本数量不一致。
    with pytest.raises(AssertionError, match="key sample_id length 3 is not equal to batch size 4"):
        DataProto.from_dict(
            tensors={"input_ids": input_ids},
            non_tensors={"sample_id": ["A", "B", "C"]},
        )


@pytest.mark.parametrize("n", [0, 16, 17])
def test_make_batch_schema(n):
    """同一个测试分别检查空批次、默认大小和不能被 4 整除的大小。"""
    data = make_batch(n)

    assert len(data) == n
    assert data.batch["input_ids"].shape == (n, 8)
    assert data.batch["input_ids"].dtype == torch.long
    assert data.batch["attention_mask"].shape == (n, 8)
    assert data.batch["attention_mask"].dtype == torch.long
    assert torch.all(data.batch["attention_mask"] == 1)
    assert data.non_tensor_batch["sample_id"].shape == (n,)
    assert data.non_tensor_batch["prompt_id"].shape == (n,)
    assert data.meta_info == {"run_id": "learning-w01", "schema_version": 1}

    # 检查每行的身份与分组；n=0 时没有行，循环不会执行。
    for i in range(n):
        assert data.batch["input_ids"][i, 0].item() == i
        assert data.non_tensor_batch["sample_id"][i] == f"s{i:03d}"
        assert data.non_tensor_batch["prompt_id"][i] == f"p{i // 4:03d}"


def test_make_batch_returns_independent_data():
    """修改一批数据，不能污染另一次调用生成的数据。"""
    first = make_batch()
    second = make_batch()

    # 分别修改张量、mask、非张量数组和元信息。
    first.batch["input_ids"][0, 0] = -1
    first.batch["attention_mask"][0, 0] = 0
    first.non_tensor_batch["sample_id"][0] = "changed"
    first.non_tensor_batch["prompt_id"][0] = "changed"
    first.meta_info["run_id"] = "changed"

    assert second.batch["input_ids"][0, 0].item() == 0
    assert second.batch["attention_mask"][0, 0].item() == 1
    assert second.non_tensor_batch["sample_id"][0] == "s000"
    assert second.non_tensor_batch["prompt_id"][0] == "p000"
    assert second.meta_info["run_id"] == "learning-w01"
