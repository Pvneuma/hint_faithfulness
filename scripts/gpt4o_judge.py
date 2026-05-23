
from utils.openai_api_framework import OpenAIHandler
import os
import sys
import json
import argparse
from openai import OpenAI
from pydantic import BaseModel, ConfigDict


class ArticulationJudged(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence: list[str]
    final_answer: bool


schema = ArticulationJudged.model_json_schema()

# 基础路径配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DATA_FILE = os.path.join(
    BASE_DIR, "output/deepseek_logic_five_results.jsonl")
PROMPT_FILE = os.path.join(BASE_DIR, "data/llm_as_judge_prompt.txt")
BATCH_INPUT_FILE = os.path.join(BASE_DIR, "data/batch_eval_input.jsonl")
BATCH_OUTPUT_FILE = os.path.join(BASE_DIR, "data/batch_eval_output.jsonl")
FINAL_MERGED_FILE = os.path.join(BASE_DIR, "output/evaluated_results.jsonl")


def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def extract_output_text(body: dict) -> str | None:

    for item in body.get("output", []):

        if item.get("type") != "message":
            continue

        for content in item.get("content", []):

            if content.get("type") == "output_text":
                return content.get("text")

    return None


def prepare_batch_data():
    prompt_template = load_prompt()

    if not os.path.exists(INPUT_DATA_FILE):
        print(f"输入文件不存在: {INPUT_DATA_FILE}")
        return

    data_list = []
    with open(INPUT_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = item["id"]

            # 分离出 ufl 和 hfl 构建独立的判定请求，通过 ID 后缀保证不混淆
            for key in ["ufl", "hfl"]:
                if key not in item:
                    continue

                model_response = item[key]
                prompt = prompt_template.replace(
                    "$model_response", model_response)

                data_list.append({
                    "custom_id": f"{row_id}_{key}",
                    "input": prompt,
                    "instructions": "You are a helpful and accurate judge.",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "ArticulationJudged",
                            "schema": schema,
                            "strict": True
                        }
                    }
                })

    handler = OpenAIHandler()
    handler.create_batch_input_file(data_list, BATCH_INPUT_FILE)
    print(f"已生成 Batch 输入文件: {BATCH_INPUT_FILE}，共 {len(data_list)} 条请求。")


def submit_batch():
    handler = OpenAIHandler()
    if not os.path.exists(BATCH_INPUT_FILE):
        print(f"找不到 Batch 输入文件: {BATCH_INPUT_FILE}，请先执行 prepare")
        return
    batch_id = handler.submit_batch_job(BATCH_INPUT_FILE)
    if batch_id:
        print(f"Batch 任务已提交成功！Batch ID: {batch_id}")
    else:
        print("Batch 任务提交失败。")


def check_status():
    handler = OpenAIHandler()
    status = handler.check_batch_status()
    if status:
        print(f"Batch 任务状态: {status.status}")
        if status.status == "completed":
            print("任务已完成，可以尝试获取结果 (retrieve)。")
        elif status.status in ["failed", "expired", "cancelled"]:
            print("任务未成功完成。")
    else:
        print("无法获取状态。")


def retrieve_and_merge():
    handler = OpenAIHandler()
    print("正在下载 Batch 结果...")
    res = handler.retrieve_batch_results(BATCH_OUTPUT_FILE)
    if not res:
        print("获取结果失败或尚未准备好。")
        return

    # 解析 batch 的结果文件，匹配 custom_id
    eval_results = {}
    with open(BATCH_OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch_res = json.loads(line)
            custom_id = batch_res.get("custom_id")

            try:
                # 提取模型最终的 JSON 回答字符串
                content_str = extract_output_text(
                    batch_res["response"]["body"])
                if content_str is None:
                    content_str = ""
                eval_data = json.loads(content_str)
            except Exception as e:
                eval_data = {"error": str(
                    e), "raw": batch_res.get("response", {})}

            eval_results[custom_id] = eval_data

    # 读取原始生成结果并合并数据
    merged_count = 0
    with open(INPUT_DATA_FILE, "r", encoding="utf-8") as f_in, \
            open(FINAL_MERGED_FILE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = item["id"]

            # 关联 gpt 对于 ufl 和 hfl 的判别回答
            item["gpt_ufl_eval"] = eval_results.get(f"{row_id}_ufl")
            item["gpt_hfl_eval"] = eval_results.get(f"{row_id}_hfl")

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            merged_count += 1

    print(f"合并完成，共处理 {merged_count} 条数据。最终文件保存在 {FINAL_MERGED_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT-4o Judge for Model Responses via Batch API")
    parser.add_argument("action", choices=[
                        "prepare", "submit", "status", "retrieve"], help="执行的操作")
    args = parser.parse_args()

    if args.action == "prepare":
        prepare_batch_data()
    elif args.action == "submit":
        submit_batch()
    elif args.action == "status":
        check_status()
    elif args.action == "retrieve":
        retrieve_and_merge()
