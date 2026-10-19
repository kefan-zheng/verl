"""观察本地 CPU 数据的共享存储、拷贝和序列化行为。"""

import pickle
from copy import deepcopy

import numpy as np
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from tests.learning_batches import make_batch
from verl import DataProto


def test_chunk_shares_data_with_original():
    """新的 DataProto 对象，不代表其中的数据已经复制。"""
    original = make_batch(16)
    first, second, _, _ = original.chunk(4)

    # 外层容器是新对象，但 meta_info 是同一个字典。
    assert first is not original
    assert first.meta_info is original.meta_info
    assert second.meta_info is original.meta_info

    # 修改张量里的元素，而不是给整个字段换一个新张量。
    first.batch["input_ids"][0, 0] = -99
    assert original.batch["input_ids"][0, 0].item() == -99

    # 非张量数组的切片也共享原数组的存储。
    first.non_tensor_batch["sample_id"][0] = "changed"
    assert original.non_tensor_batch["sample_id"][0] == "changed"

    # 各块的样本区间不重叠，因此第一块的行修改不会改到第二块的行。
    assert second.batch["input_ids"][0, 0].item() == 4
    assert second.non_tensor_batch["sample_id"][0] == "s004"

    # 但共享的元信息字典被修改时，原批次和所有子批次都会看到变化。
    first.meta_info["run_id"] = "changed"
    assert original.meta_info["run_id"] == "changed"
    assert second.meta_info["run_id"] == "changed"


def test_deepcopy_chunk_isolates_mutations():
    """对这个 CPU 合成批次深拷贝后，修改副本不会污染原批次。"""
    original = make_batch(16)
    shared_part = original.chunk(4)[0]
    independent_part = deepcopy(shared_part)

    assert independent_part is not shared_part
    assert independent_part.meta_info is not original.meta_info

    independent_part.batch["input_ids"][0, 0] = -99
    independent_part.non_tensor_batch["sample_id"][0] = "changed"
    independent_part.meta_info["run_id"] = "changed"

    # 两个未修改对象都应该保留原值。
    for data in (original, shared_part):
        assert data.batch["input_ids"][0, 0].item() == 0
        assert data.non_tensor_batch["sample_id"][0] == "s000"
        assert data.meta_info["run_id"] == "learning-w01"


def test_pickle_roundtrip_preserves_data_and_isolates_mutations():
    """只序列化自己构造的数据，检查 int64、float32、bool 的往返。"""
    original = make_batch(4)
    # input_ids 已经是 int64，再加入两种类型。
    original.batch["scores"] = torch.tensor([0.25, -0.5, 1.0, 2.0], dtype=torch.float32)
    original.batch["valid"] = torch.tensor([True, False, True, False], dtype=torch.bool)

    # dumps 返回 bytes，loads 从 bytes 重建对象；整个过程发生在内存中。
    # pickle 会调用 DataProto 定义的 __getstate__ / __setstate__。
    payload = pickle.dumps(original)
    assert isinstance(payload, bytes)
    assert len(payload) > 0
    restored = pickle.loads(payload)

    assert isinstance(restored, DataProto)
    assert restored is not original
    assert len(restored) == len(original)
    assert set(restored.batch.keys()) == set(original.batch.keys())
    assert set(restored.non_tensor_batch) == set(original.non_tensor_batch)

    # 值相等还不够：dtype 和 shape 也应该保留。
    for key in original.batch.keys():
        assert restored.batch[key].dtype == original.batch[key].dtype
        assert restored.batch[key].shape == original.batch[key].shape
        assert torch.equal(restored.batch[key], original.batch[key])
    for key in original.non_tensor_batch:
        assert restored.non_tensor_batch[key].dtype == original.non_tensor_batch[key].dtype
        assert np.array_equal(restored.non_tensor_batch[key], original.non_tensor_batch[key])
    assert restored.meta_info == original.meta_info

    # 验证本次还原的数据与原数据独立，不能仅凭 is not 判断存储独立。
    restored.batch["input_ids"][0, 0] = -99
    restored.batch["scores"][0] = 9.0
    restored.batch["valid"][0] = False
    restored.non_tensor_batch["sample_id"][0] = "changed"
    restored.meta_info["run_id"] = "changed"

    assert original.batch["input_ids"][0, 0].item() == 0
    assert original.batch["scores"][0].item() == 0.25
    assert original.batch["valid"][0].item() is True
    assert original.non_tensor_batch["sample_id"][0] == "s000"
    assert original.meta_info["run_id"] == "learning-w01"


def test_tensordict_roundtrip_preserves_field_categories():
    """三类字段合入 TensorDict，再转换回 DataProto 的三类成员。"""
    original = make_batch(4)
    original.batch["scores"] = torch.tensor([0.25, -0.5, 1.0, 2.0], dtype=torch.float32)
    original.batch["valid"] = torch.tensor([True, False, True, False], dtype=torch.bool)

    converted = original.to_tensordict()
    assert isinstance(converted, TensorDict)
    assert converted.batch_size == torch.Size([4])

    # 转换后的字段位于同一个容器中；非张量通过不同包装区分语义。
    # items() 能看到包装对象，普通 [] 访问某些字段会自动解包。
    fields = dict(converted.items())
    expected_keys = set(original.batch.keys()) | set(original.non_tensor_batch) | set(original.meta_info)
    assert set(fields) == expected_keys
    assert isinstance(fields["input_ids"], torch.Tensor)
    assert isinstance(fields["sample_id"], NonTensorStack)
    assert isinstance(fields["run_id"], NonTensorData)

    restored = DataProto.from_tensordict(converted)
    assert len(restored) == 4
    assert set(restored.batch.keys()) == set(original.batch.keys())
    assert set(restored.non_tensor_batch) == set(original.non_tensor_batch)
    for key in original.batch.keys():
        assert restored.batch[key].shape == original.batch[key].shape
        assert restored.batch[key].dtype == original.batch[key].dtype
        assert torch.equal(restored.batch[key], original.batch[key])
    for key in original.non_tensor_batch:
        assert restored.non_tensor_batch[key].dtype == original.non_tensor_batch[key].dtype
        assert np.array_equal(restored.non_tensor_batch[key], original.non_tensor_batch[key])
    assert restored.meta_info == original.meta_info


def test_tensordict_conversion_keeps_shared_tensor_storage():
    """容器转换不等于深拷贝，也不同于前面的 pickle 字节往返。"""
    original = make_batch(4)
    converted = original.to_tensordict()
    restored = DataProto.from_tensordict(converted)

    assert restored is not original
    restored.batch["input_ids"][0, 0] = -99

    # 当前转换复用张量；从其中一个容器修改元素，其他容器也能看到。
    assert converted["input_ids"][0, 0].item() == -99
    assert original.batch["input_ids"][0, 0].item() == -99
