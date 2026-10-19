"""从仓库根目录运行（单机复用现有环境，避免 Ray 在 /tmp 重复安装依赖）：

RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --frozen --all-packages --extra cpu \
python -m examples.learning_w04_ray_roundtrip
"""

import os
from copy import deepcopy

import numpy as np
import ray
import torch

from tests.learning_batches import make_batch
from verl import DataProto


def add_one(data: DataProto) -> DataProto:
    """最小处理逻辑：返回独立结果，仅给 input_ids 加 1。"""
    result = deepcopy(data)
    result.batch["input_ids"] = result.batch["input_ids"] + 1
    return result


@ray.remote(num_cpus=1)
class BatchWorker:
    """每个 actor 在独立进程里执行方法。"""

    def __init__(self):
        torch.set_num_threads(1)

    def process(self, shard_id: int, data: DataProto):
        return {"shard_id": shard_id, "pid": os.getpid(), "data": add_one(data)}


def assert_same_data(actual: DataProto, expected: DataProto):
    assert len(actual) == len(expected)
    assert set(actual.batch.keys()) == set(expected.batch.keys())
    assert set(actual.non_tensor_batch) == set(expected.non_tensor_batch)
    for key in expected.batch.keys():
        assert actual.batch[key].shape == expected.batch[key].shape
        assert actual.batch[key].dtype == expected.batch[key].dtype
        assert torch.equal(actual.batch[key], expected.batch[key]), key
    for key in expected.non_tensor_batch:
        assert actual.non_tensor_batch[key].dtype == expected.non_tensor_batch[key].dtype
        assert np.array_equal(actual.non_tensor_batch[key], expected.non_tensor_batch[key]), key
    assert actual.meta_info == expected.meta_info


def main():
    torch.set_num_threads(1)
    original = make_batch(16)
    snapshot = deepcopy(original)
    expected = add_one(original)  # 先在主进程计算参考结果。
    assert torch.equal(expected.batch["input_ids"], snapshot.batch["input_ids"] + 1)
    assert_same_data(original, snapshot)

    # 只启动本脚本自己的本地会话，2 个逻辑 CPU、不使用 GPU。
    # 按文件顶部命令关闭 uv 自动环境复制，让本机 worker 使用当前 Python 环境。
    if ray.is_initialized():
        raise RuntimeError("请在独立 Python 进程中运行此学习脚本")
    try:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=0,
            include_dashboard=False,
            object_store_memory=128 * 1024 * 1024,
        )
        workers = [BatchWorker.remote(), BatchWorker.remote()]
        parts = original.chunk(2)

        # remote() 提交工作，立即返回 ObjectRef，而不是最终结果。
        refs = [workers[i].process.remote(i, parts[i]) for i in range(2)]
        results = ray.get(refs, timeout=90)
        # ray.get(list) 的结果对应输入引用顺序，不是 worker 完成顺序。
        assert [result["shard_id"] for result in results] == [0, 1]
        worker_pids = {result["pid"] for result in results}
        assert len(worker_pids) == 2
        assert os.getpid() not in worker_pids

        merged = DataProto.concat([result["data"] for result in results])
        assert_same_data(merged, expected)
        assert_same_data(original, snapshot)

        # 再提交一轮任务，对比逐个收集已就绪结果的方式。
        # 不能先 ray.get 全部结果再演示 wait，否则已经失去等待过程。
        pending = [workers[i].process.remote(i, parts[i]) for i in range(2)]
        observed_results = []
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=90)
            if not ready:
                raise TimeoutError("90 秒内没有分片结果就绪")
            # wait 返回引用；get 才取回 process() 返回的字典。
            result = ray.get(ready[0], timeout=90)
            observed_results.append(result)
            print(f"ray.wait 收到分片 {result['shard_id']}，尚待收集 {len(pending)} 份")

        # 观测到的就绪顺序不保证是 [0, 1]，也不保证是 [1, 0]。
        # 同时有多个结果就绪时，wait 不能作为精确完成时间的排序器。
        observed_order = [result["shard_id"] for result in observed_results]
        assert sorted(observed_order) == [0, 1]  # 没有丢失或重复收集分片。
        ordered_results = sorted(observed_results, key=lambda result: result["shard_id"])
        merged_after_wait = DataProto.concat([result["data"] for result in ordered_results])
        assert_same_data(merged_after_wait, expected)
        assert_same_data(original, snapshot)

        print(f"driver PID: {os.getpid()}")
        for result in results:
            ids = result["data"].non_tensor_batch["sample_id"].tolist()
            print(f"shard {result['shard_id']}: worker PID={result['pid']}, samples={ids}")
        print(f"ray.get 列表返回顺序: {[result['shard_id'] for result in results]}")
        print(f"ray.wait 本次观测顺序: {observed_order}")
        print("PASS: get 与 wait 两种收集方式均匹配本地参考，16 条样本顺序正确，原数据未被修改。")
    finally:
        ray.shutdown()
        print("Ray 会话已关闭。")


if __name__ == "__main__":
    main()
