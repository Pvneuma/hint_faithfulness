from utils.openai_api_framework import OpenAIHandler
import os
import sys
import json
import time
import argparse

# 基础路径配置
INPUT_DATA_FILE = "output/qwen3_logic_five_valuated_results.jsonl"
PROMPT_FILE = "data/extract_answer_prompt.txt"
BATCH_OUTPUT_FILE = "data/batch_extract_output.jsonl"
FINAL_MERGED_FILE = "output/qwen3_logic_five_extracted_results.jsonl"
BATCH_RECORD_FILE = "data/completed_extract_batches.jsonl"


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


def process_batches():
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

            # 分别为 ans, ufl, hfl 构建提取任务
            for key in ["ans", "ufl", "hfl"]:
                if key not in item:
                    continue

                model_response = item[key]
                if not isinstance(model_response, str):
                    model_response = str(model_response)

                prompt = prompt_template.replace("$response", model_response)

                data_list.append({
                    "custom_id": f"{row_id}_{key}",
                    "input": prompt,
                })

    handler = OpenAIHandler()

    # 每次开始前清空记录文件
    with open(BATCH_RECORD_FILE, "w", encoding="utf-8") as f:
        pass

    chunk_size = 10  # 根据实际数据量可以适当调整
    for i in range(0, len(data_list), chunk_size):
        chunk = data_list[i:i + chunk_size]
        chunk_file = "data/batch_extract_input.jsonl"

        # 使用 gpt-4o-mini-2024-07-18 并将 temperature 设置为 0.0 以获取稳定的提取结果
        handler.create_batch_input_file(
            chunk, chunk_file, model="gpt-4o-mini-2024-07-18", temperature=0.0)
        print(f"提交第 {i//chunk_size + 1} 批任务，包含 {len(chunk)} 条数据...")

        batch_id = handler.submit_batch_job(chunk_file)
        if not batch_id:
            print("提交失败，退出。")
            return

        print(f"Batch 任务已提交，ID: {batch_id}，正在等待完成...")

        while True:
            status = handler.check_batch_status(batch_id)
            if status:
                print(f"当前状态: {status.status}")
                if status.status == "completed":
                    print(f"Batch {batch_id} 已完成。")
                    with open(BATCH_RECORD_FILE, "a", encoding="utf-8") as f_rec:
                        f_rec.write(json.dumps({"batch_id": batch_id}) + "\n")
                    break
                elif status.status in ["failed", "expired", "cancelled"]:
                    print(f"Batch {batch_id} 失败，状态: {status.status}。退出。")
                    return
            else:
                print("查询状态失败。")

            time.sleep(15)

    print("所有批次处理完成，可以使用 retrieve 命令下载并合并结果。")


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
    print("正在下载所有 Batch 结果...")
    if not os.path.exists(BATCH_RECORD_FILE):
        print(f"找不到记录文件: {BATCH_RECORD_FILE}")
        return

    handler.retrieve_batch_batch_results(BATCH_OUTPUT_FILE, BATCH_RECORD_FILE)
    if not os.path.exists(BATCH_OUTPUT_FILE) or os.path.getsize(BATCH_OUTPUT_FILE) == 0:
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
                # 提取模型最终的回答字符串
                content_str = extract_output_text(
                    batch_res["response"]["body"])
                if content_str is None:
                    content_str = "X"
                else:
                    content_str = content_str.strip().upper()
                    if content_str not in ["A", "B", "C", "D", "E", "Z"]:
                        content_str = "X"
            except Exception as e:
                print(f"解析 custom_id {custom_id} 失败: {e}")
                content_str = "X"

            eval_results[custom_id] = content_str

    # 读取原始生成结果并合并数据
    merged_count = 0
    with open(INPUT_DATA_FILE, "r", encoding="utf-8") as f_in, \
            open(FINAL_MERGED_FILE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = item["id"]

            # 关联提取出的答案，如果没有提取到默认X
            item["extracted_ans"] = eval_results.get(f"{row_id}_ans", "X")
            item["extracted_ufl"] = eval_results.get(f"{row_id}_ufl", "X")
            item["extracted_hfl"] = eval_results.get(f"{row_id}_hfl", "X")

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            merged_count += 1

    print(f"合并完成，共处理 {merged_count} 条数据。最终文件保存在 {FINAL_MERGED_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Final Answers via Batch API")
    parser.add_argument("action", choices=[
                        "process", "status", "retrieve"], help="执行的操作")
    args = parser.parse_args()

    if args.action == "process":
        process_batches()
    elif args.action == "status":
        check_status()
    elif args.action == "retrieve":
        retrieve_and_merge()
