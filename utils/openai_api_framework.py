import os
import yaml
import json
from typing import List, Dict, Optional, Any
from openai import OpenAI, OpenAIError


class OpenAIHandler:
    def __init__(self):
        self.batch_id_record_path = "data/batch_id_record.txt"
        api_key = None

        if os.path.exists("config.yml"):
            try:
                with open("config.yml", "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config:
                        # 尝试读取 openai_api_key 或 openai.api_key
                        api_key = config.get("openai_api_key")
                        print(api_key)
            except Exception as e:
                print(f"读取 config.yml 失败: {e}")

        if not api_key:
            raise ValueError("未在 config.yml 中找到有效的 API Key")

        self.client = OpenAI(
            api_key=api_key,
        )

    def create_batch_input_file(self,
                                data_list: List[Dict[str, Any]],
                                output_file_path: str,
                                model: str = "gpt-4o-2024-08-06",
                                temperature: Optional[float] = None,):
        """
        辅助方法：将数据列表转换为 Batch API 需要的 JSONL 格式。
        Endpoint: /v1/responses
        """
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item in data_list:
                # 构造 /v1/responses 的请求体
                body = {
                    "model": model,
                    "input": item.get("input"),
                    "instructions": item.get("instructions", "")
                }

                if item.get("text") is not None:
                    body["text"] = item.get("text")

                if temperature is not None:
                    body["temperature"] = temperature

                batch_request = {
                    "custom_id": str(item.get("custom_id", "")),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body
                }
                f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")

    def submit_batch_job(self, jsonl_file_path: str) -> Optional[str]:
        """上传文件并提交 Batch 任务"""
        try:
            # 1. 上传文件
            with open(jsonl_file_path, "rb") as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose="batch"
                )

            # 2. 创建 Batch 任务
            batch_response = self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/responses",
                completion_window="24h"
            )

            # 持久化 batch_id 到本地文件
            with open(self.batch_id_record_path, "w", encoding="utf-8") as f:
                f.write(f"{batch_response.id}\n")

            return batch_response.id
        except OpenAIError as e:
            print(f"OpenAI API 请求错误: {e}")
            return None

    def check_batch_status(self, batch_id: Optional[str] = None) -> Any:
        """查询 Batch 任务状态"""
        if not batch_id and os.path.exists(self.batch_id_record_path):
            with open(self.batch_id_record_path, "r", encoding="utf-8") as f:
                batch_id = f.read().strip()

        if not batch_id:
            print("未指定 Batch ID 且本地未找到记录")
            return None

        try:
            return self.client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"查询状态失败: {e}")
            return None

    def retrieve_batch_results(self, output_file_path: str, batch_id: Optional[str] = None) -> Optional[str]:
        """下载 Batch 结果并保存到文件"""
        if not batch_id and os.path.exists(self.batch_id_record_path):
            with open(self.batch_id_record_path, "r", encoding="utf-8") as f:
                batch_id = f.read().strip()

        try:
            if not batch_id:
                raise ValueError("未提供 Batch ID 且本地无记录")

            batch = self.client.batches.retrieve(batch_id)
            if not batch.output_file_id:
                print(
                    f"Batch 任务 {batch_id} 尚未生成 output_file_id (状态: {batch.status})")
                return None

            content = self.client.files.content(batch.output_file_id).text

            # 逐行解析 JSON 并使用 ensure_ascii=False 重新序列化，以修复中文转义问题
            decoded_lines = []
            for line in content.splitlines():
                if line.strip():
                    decoded_lines.append(json.dumps(
                        json.loads(line), ensure_ascii=False))
            final_content = "\n".join(decoded_lines)

            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(final_content)
            return final_content
        except Exception as e:
            print(f"下载结果失败: {e}")
            return None

    def retrieve_batch_batch_results(self, output_file_path: str, batch_id_file_path: str) -> Optional[str]:
        """下载 Batch 结果并保存到文件"""
        with open(batch_id_file_path, "r", encoding="utf-8") as f:
            completed_batch_id_list = [
                json.loads(line)['batch_id'] for line in f]
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                for batch_id in completed_batch_id_list:
                    batch = self.client.batches.retrieve(batch_id)
                    if not batch.output_file_id:
                        print(
                            f"Batch 任务 {batch_id} 尚未生成 output_file_id (状态: {batch.status})")
                        return None
                    content = self.client.files.content(
                        batch.output_file_id).text
                    decoded_lines = []
                    for line in content.splitlines():
                        if line.strip():
                            decoded_lines.append(json.dumps(
                                json.loads(line), ensure_ascii=False))
                    if decoded_lines:
                        final_content = "\n".join(decoded_lines)
                        f.write(final_content + "\n")
        except Exception as e:
            print(f"下载结果失败: {e}")
            return None
