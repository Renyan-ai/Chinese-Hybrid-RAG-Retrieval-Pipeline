from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision

TEST_SET = [
    {"question": "2025年腾讯全年总收入是多少", "ground_truth": "总收入为人民币7,518亿元"},
    {"question": "2025年腾讯毛利是多少", "ground_truth": "毛利为人民币4,226亿元"},
    {"question": "2025年第四季度总收入是多少", "ground_truth": "总收入为人民币1,944亿元"},
    {"question": "2025年增值服务业务收入", "ground_truth": "增值服务业务收入同比增长16%至人民币3,693亿元"},
    {"question": "2025年营销服务收入", "ground_truth": "营销服务收入同比增长19%至人民币1,450亿元"},
    {"question": "微信月活跃用户数", "ground_truth": "微信及WeChat的合并月活跃账户数为1,418百万"},
    {"question": "腾讯2025年业务回顾中提到了哪些游戏", "ground_truth": "三角洲行动、王者荣耀、和平精英"},
]


def run_ragas_eval(retriever, test_set=None, llm=None):
    if test_set is None:
        test_set = TEST_SET

    print(f"\n{'='*60}")
    print(" RAGAS 检索精度评估")
    print(f"{'='*60}")

    ragas_inputs = {"question": [], "contexts": [], "ground_truth": []}

    for item in test_set:
        q = item["question"]
        print(f"\n 正在检索: 「{q}」")
        ctx_list = retriever.retrieve_contexts(q)
        ragas_inputs["question"].append(q)
        ragas_inputs["contexts"].append(ctx_list)
        ragas_inputs["ground_truth"].append(item["ground_truth"])
        print(f"   检索到 {len(ctx_list)} 个上下文片段")

    dataset = Dataset.from_dict(ragas_inputs)

    print(f"\n{'='*60}")
    print("  正在运行 RAGAS 评分...")
    print(f"{'='*60}")

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision],
        llm=llm,
    )

    print(f"\n{'='*60}")
    print(" 评估结果")
    print(f"{'='*60}")

    print(f"\n{'─'*60}")
    print(" 逐问题 precision:")
    scores = result['context_precision']
    for i, item in enumerate(test_set):
        s = scores[i] if isinstance(scores, (list, tuple)) and i < len(scores) else "-"
        print(f"  [{i+1}] {item['question']}")
        print(f"       正确答案: {item['ground_truth']}")
        print(f"       precision: {s}")

    return result
