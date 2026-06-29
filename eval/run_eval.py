#!/usr/bin/env python3
"""
RAG 评估运行器
加载测试集 → 通过 HTTP API 查询系统 → 评估指标 → 输出报告

用法:
    conda run -n kbqa python eval/run_eval.py
    conda run -n kbqa python eval/run_eval.py --top-k 3 --max-cases 10
    conda run -n kbqa python eval/run_eval.py --type reject
"""
import sys
import os
import json
import time
import argparse
import httpx
from typing import List, Dict
from collections import defaultdict

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import evaluate_case, EvalResult
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("eval")

TESTSET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "eval_testset.json"
)

API_BASE = "http://localhost:8000/api"


def load_testset(path: str = TESTSET_PATH) -> List[Dict]:
    """加载测试集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_query(query: str, top_k: int = 5) -> tuple:
    """
    通过 HTTP API 运行单条查询，返回 (answer, retrieved_docs)
    
    调用 POST /chat 获取 answer + references
    """
    client = httpx.Client(timeout=120.0)
    
    try:
        resp = client.post(f"{API_BASE}/chat", json={
            "query": query,
            "stream": False,
        })
        resp.raise_for_status()
        data = resp.json()
        
        answer = data.get("answer", "")
        references = data.get("references", [])
        
        # references 格式与检索结果一致，直接作为 docs
        docs = references if references else []
        
        return answer, docs
    finally:
        client.close()


def load_testset(path: str = TESTSET_PATH) -> List[Dict]:
    """加载测试集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_query(query: str, top_k: int = 5) -> tuple:
    """
    通过 HTTP API 运行单条查询，返回 (answer, retrieved_docs)
    
    调用 POST /chat 获取 answer + references
    """
    client = httpx.Client(timeout=120.0)
    
    try:
        resp = client.post(f"{API_BASE}/chat", json={
            "query": query,
            "stream": False,
        })
        resp.raise_for_status()
        data = resp.json()
        
        answer = data.get("answer", "")
        references = data.get("references", [])
        
        # references 格式与检索结果一致，直接作为 docs
        docs = references if references else []
        
        return answer, docs
    finally:
        client.close()


def print_summary(results: List[EvalResult], elapsed: float):
    """打印评估摘要报告"""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    
    # 按类型分组
    by_type = defaultdict(list)
    for r in results:
        by_type[r.test_type].append(r)
    
    # 按难度分组
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r.difficulty].append(r)
    
    print("\n" + "=" * 70)
    print("RAG 评估报告")
    print("=" * 70)
    print(f"总用例: {total}  通过: {passed}  通过率: {passed/total*100:.1f}%")
    print(f"总耗时: {elapsed:.1f}s  平均: {elapsed/total:.1f}s/条")
    
    # 按类型
    print(f"\n{'类型':<12} {'总数':>4} {'通过':>4} {'通过率':>8} {'关键词召回':>10} {'来源召回':>10}")
    print("-" * 52)
    for t, cases in sorted(by_type.items()):
        t_pass = sum(1 for c in cases if c.passed)
        t_kw = sum(c.keyword_recall for c in cases) / len(cases)
        t_src = sum(c.source_recall for c in cases) / len(cases)
        print(f"{t:<12} {len(cases):>4} {t_pass:>4} {t_pass/len(cases)*100:>7.1f}% {t_kw:>10.2f} {t_src:>10.2f}")
    
    # 按难度
    print(f"\n{'难度':<12} {'总数':>4} {'通过':>4} {'通过率':>8}")
    print("-" * 32)
    for d in ["easy", "medium", "hard"]:
        if d in by_diff:
            cases = by_diff[d]
            d_pass = sum(1 for c in cases if c.passed)
            print(f"{d:<12} {len(cases):>4} {d_pass:>4} {d_pass/len(cases)*100:>7.1f}%")
    
    # 失败用例详情
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n{'─' * 70}")
        print(f"失败用例 ({len(failed)} 条):")
        print(f"{'─' * 70}")
        for r in failed:
            print(f"\n  [{r.id}] {r.query}")
            print(f"  类型={r.test_type}, 难度={r.difficulty}")
            if r.test_type == "reject":
                print(f"  期望: 拒答  实际: {'正确拒答' if r.reject_correct else '未拒答 ❌'}")
            else:
                print(f"  关键词召回: {r.keyword_recall:.2f} ({len(r.keywords_found)}/{len(r.keywords_found)+len(r.keywords_missing)})")
                if r.keywords_missing:
                    print(f"    缺失: {', '.join(r.keywords_missing[:5])}")
                print(f"  来源召回: {r.source_recall:.2f} ({len(r.sources_found)}/{len(r.sources_found)+len(r.sources_missing)})")
                if r.sources_missing:
                    print(f"    缺失: {', '.join(r.sources_missing)}")
            # 截断答案显示
            answer_preview = r.answer[:150].replace("\n", " ")
            print(f"  答案预览: {answer_preview}...")
    
    print("\n" + "=" * 70)
    
    # 保存详细结果到文件
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "eval_report.json"
    )
    report = {
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total * 100, 1),
            "elapsed_seconds": round(elapsed, 1),
            "by_type": {
                t: {
                    "total": len(cases),
                    "passed": sum(1 for c in cases if c.passed),
                    "avg_keyword_recall": round(sum(c.keyword_recall for c in cases) / len(cases), 3),
                    "avg_source_recall": round(sum(c.source_recall for c in cases) / len(cases), 3),
                }
                for t, cases in by_type.items()
            },
        },
        "results": [
            {
                "id": r.id,
                "query": r.query,
                "type": r.test_type,
                "difficulty": r.difficulty,
                "passed": r.passed,
                "keyword_recall": round(r.keyword_recall, 3),
                "source_recall": round(r.source_recall, 3),
                "reject_correct": r.reject_correct,
                "keywords_missing": r.keywords_missing,
                "sources_missing": r.sources_missing,
                "answer_preview": r.answer[:300],
            }
            for r in results
        ],
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"详细报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="RAG 评估运行器")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回文档数 (默认 5)")
    parser.add_argument("--max-cases", type=int, default=0, help="最多运行 N 条 (0=全部)")
    parser.add_argument("--type", type=str, default="", help="只运行指定类型 (factual/exact_match/conceptual/reject/cross_doc)")
    parser.add_argument("--difficulty", type=str, default="", help="只运行指定难度 (easy/medium/hard)")
    parser.add_argument("--testset", type=str, default=TESTSET_PATH, help="测试集路径")
    args = parser.parse_args()
    
    # 加载测试集
    testset = load_testset(args.testset)
    if args.type:
        testset = [t for t in testset if t["type"] == args.type]
    if args.difficulty:
        testset = [t for t in testset if t["difficulty"] == args.difficulty]
    if args.max_cases > 0:
        testset = testset[:args.max_cases]
    
    print(f"加载测试集: {len(testset)} 条用例")
    print(f"配置: top_k={args.top_k}, max_context_tokens={settings.max_context_tokens}")
    print(f"Reranker: {'启用' if settings.reranker_enabled else '禁用'} ({settings.reranker_model})")
    print(f"API: {API_BASE}")
    print()
    
    results = []
    start_time = time.time()
    
    for i, test_case in enumerate(testset, 1):
        query = test_case["query"]
        tc_id = test_case["id"]
        tc_type = test_case["type"]
        
        print(f"[{i}/{len(testset)}] #{tc_id} [{tc_type}] {query[:50]}...", end=" ", flush=True)
        
        try:
            t0 = time.time()
            answer, docs = run_single_query(query, top_k=args.top_k)
            t1 = time.time()
            
            result = evaluate_case(test_case, answer, docs)
            results.append(result)
            
            status = "✓" if result.passed else "❌"
            kw_str = f"kw={result.keyword_recall:.2f}" if tc_type != "reject" else ""
            src_str = f"src={result.source_recall:.2f}" if tc_type != "reject" else ""
            print(f"{status} {kw_str} {src_str} ({t1-t0:.1f}s)")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            logger.error(f"测试用例 #{tc_id} 执行失败: {e}")
            results.append(EvalResult(
                id=tc_id, query=query, test_type=tc_type,
                difficulty=test_case.get("difficulty", "medium"),
            ))
    
    elapsed = time.time() - start_time
    print_summary(results, elapsed)


if __name__ == "__main__":
    main()
