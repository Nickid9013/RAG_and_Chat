import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param 
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_fixed
import dashscope

# [当前默认流程未使用] 仅当 api_provider="openai" 时使用，默认 dashscope
# OpenAI基础处理器，封装了消息发送、结构化输出、计费等逻辑
class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        # 加载OpenAI API密钥，初始化LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    HISTORY_MAX_MESSAGES = 8

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None, # For deterministic ouptputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        history=None,
        **kwargs
        ):
        # 发送消息到OpenAI，支持结构化/非结构化输出；支持多轮 history，仅保留最近 HISTORY_MAX_MESSAGES 条
        if model is None:
            model = self.default_model
        messages = [{"role": "system", "content": system_content}]
        if history:
            trimmed = history[-self.HISTORY_MAX_MESSAGES:] if len(history) > self.HISTORY_MAX_MESSAGES else history
            messages.extend(trimmed)
        messages.append({"role": "user", "content": human_content})
        params = {
            "model": model,
            "seed": seed,
            "messages": messages,
        }
        
        # 部分模型不支持temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature
            
        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens}
        print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        # 统计字符串的token数
        encoding = tiktoken.get_encoding(encoding_name)
        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count


# [当前流程未使用] 仅当 api_provider="ibm" 时使用，默认 dashscope
# IBM API基础处理器，支持余额查询、模型列表、嵌入、消息发送等
class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """查询当前API余额"""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """获取可用基础模型列表"""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """获取文本的向量嵌入"""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        # 发送消息到IBM API，支持结构化/非结构化输出
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

# [当前流程未使用] 仅当 api_provider="gemini" 时使用，默认 dashscope
class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 20 seconds before retry...\n")

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Wrapper for generate_content with retry logic"""
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")

# -----------------------------------------------------------------------------
# DashScope (legacy) - 已注释保留，便于回滚对照
# -----------------------------------------------------------------------------
# DashScope基础处理器，支持Qwen大模型对话
# class BaseDashscopeProcessor:
#     def __init__(self):
#         # 从环境变量读取API-KEY
#         dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
#         self.default_model = 'qwen3.5-plus'

#     def send_message(
#         self,
#         model="qwen3.5-plus",
#         temperature=0.1,
#         seed=None,  # 兼容参数，暂不使用
#         system_content='You are a helpful assistant.',
#         human_content='Hello!',
#         is_structured=False,
#         response_format=None,
#         **kwargs
#     ):
#         """
#         发送消息到DashScope Qwen大模型，支持 system_content + human_content 拼接为 messages。
#         暂不支持结构化输出。
#         """
#         if model is None:
#             model = self.default_model
#         # 拼接 messages
#         messages = []
#         if system_content:
#             messages.append({"role": "system", "content": system_content})
#         if human_content:
#             messages.append({"role": "user", "content": human_content})
#         #print('system_content=', system_content)
#         #print('='*30)
#         #print('human_content=', human_content)
#         #print('='*30)
#         #print('messages=', messages)
#         #print('='*30)
#         # 调用 dashscope Generation.call
#         response = dashscope.Generation.call(
#             model=model,
#             messages=messages,
#             temperature=temperature,
#             result_format='message'
#         )
#         print('dashscope.api_key=', dashscope.api_key)
#         print('model=', model)
#         print('response=', response)
#         # 兼容 openai/gemini 返回格式，始终返回 dict
#         if hasattr(response, 'output') and hasattr(response.output, 'choices'):
#             content = response.output.choices[0].message.content
#         else:
#             content = str(response)
#         # 增加 response_data 属性，保证接口一致性
#         self.response_data = {"model": model, "input_tokens": response.usage.input_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens') else None, "output_tokens": response.usage.output_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens') else None}
#         print('content=', content)
        
#         # 尝试解析 content 为 JSON，如果是结构化响应
#         try:
#             # 先尝试移除可能的markdown代码块标记
#             content_str = content.strip()
#             if content_str.startswith('```') and '```' in content_str[3:]:
#                 # 找到第一个 ``` 和 最后一个 ``` 之间的内容
#                 first_backtick = content_str.find('```') + 3
#                 next_newline = content_str.find('\n', first_backtick)
#                 if next_newline > 0:
#                     first_backtick = next_newline + 1
#                 last_backtick = content_str.rfind('```')
#                 if last_backtick > first_backtick:
#                     json_str = content_str[first_backtick:last_backtick].strip()
#                 else:
#                     json_str = content_str
#             else:
#                 json_str = content_str
            
#             # 尝试解析 JSON
#             parsed_content = json.loads(json_str)
#             return parsed_content
#         except (json.JSONDecodeError, TypeError):
#             # 如果不是有效的JSON，返回基本格式
#             print(f"Content is not valid JSON, returning basic format: {content}")
#             return {"final_answer": content, "step_by_step_analysis": "", "reasoning_summary": "", "relevant_sources": []}


# DashScope 基础处理器（新版）：支持基于 Pydantic schema 的结构化输出与校验
class BaseDashscopeProcessor:
    def __init__(self):
        # 从环境变量读取 API-KEY
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.default_model = "qwen3.5-plus"
        self.response_data = {}

    def _call_generation(self, model: str, messages: List[Dict[str, str]], temperature: float):
        return dashscope.Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            result_format="message",
        )

    def _extract_content(self, response) -> str:
        # DashScope SDK 有时是对象，有时是 dict；这里尽量兼容
        try:
            if isinstance(response, dict) and "output" in response and "choices" in response["output"]:
                return response["output"]["choices"][0]["message"]["content"]
        except Exception:
            pass
        if hasattr(response, "output") and hasattr(response.output, "choices"):
            return response.output.choices[0].message.content
        return str(response)

    def _extract_json_str(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        content_str = text.strip()
        # 去掉 ```json ... ``` 包裹
        if content_str.startswith("```") and "```" in content_str[3:]:
            first_backtick = content_str.find("```") + 3
            next_newline = content_str.find("\n", first_backtick)
            if next_newline > 0:
                first_backtick = next_newline + 1
            last_backtick = content_str.rfind("```")
            if last_backtick > first_backtick:
                return content_str[first_backtick:last_backtick].strip()
        # 尝试截取第一个 {...}（防止带前后解释）
        first = content_str.find("{")
        last = content_str.rfind("}")
        if 0 <= first < last:
            return content_str[first:last + 1].strip()
        return content_str

    def _structured_system_content(self, system_content: str, response_format: Type[BaseModel]) -> str:
        schema_json = json.dumps(response_format.model_json_schema(), ensure_ascii=False, indent=2)
        return (
            (system_content or "").rstrip()
            + "\n\n你必须只输出合法JSON（不要输出多余文字、不要使用Markdown代码块）。"
            + "\n请严格满足如下 JSON Schema：\n```json\n"
            + schema_json
            + "\n```"
        )

    def _parse_structured(self, content: str, response_format: Type[BaseModel]) -> Dict:
        json_str = self._extract_json_str(content)
        try:
            parsed = json.loads(json_str)
        except Exception:
            # 使用 json_repair 做一次容错修复
            parsed = json.loads(repair_json(json_str))
        validated = response_format.model_validate(parsed)
        return validated.model_dump()

    def _reparse_with_fix_prompt(self, system_content: str, response_text: str, model: str, temperature: float) -> str:
        # 用 AnswerSchemaFixPrompt 将“非 JSON/不合法 JSON”修复为合法 JSON
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response_text,
        )
        messages = [
            {"role": "system", "content": prompts.AnswerSchemaFixPrompt.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        fix_rsp = self._call_generation(model=model, messages=messages, temperature=0)
        return self._extract_content(fix_rsp)

    # 多轮对话上下文最大条数（user+assistant 合计），超过则丢弃最早的消息
    HISTORY_MAX_MESSAGES = 8

    def send_message(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        seed=None,  # 兼容参数，暂不使用
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        if model is None:
            model = self.default_model

        effective_system = system_content
        if is_structured and response_format is not None:
            effective_system = self._structured_system_content(system_content, response_format)

        messages: List[Dict[str, str]] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        # 短期对话上下文：只保留最近 HISTORY_MAX_MESSAGES 条，超过部分直接删除
        if history:
            trimmed = history[-self.HISTORY_MAX_MESSAGES:] if len(history) > self.HISTORY_MAX_MESSAGES else history
            messages.extend(trimmed)
        if human_content:
            messages.append({"role": "user", "content": human_content})

        response = self._call_generation(model=model, messages=messages, temperature=temperature)
        content = self._extract_content(response)

        # print("dashscope.api_key=", dashscope.api_key)
        # print("model=", model)
        # print("response=", response)

        # response_data：尽量提供 usage 信息，保持接口一致性
        input_tokens = None
        output_tokens = None
        try:
            if isinstance(response, dict) and "usage" in response:
                usage = response["usage"] or {}
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
            elif hasattr(response, "usage"):
                input_tokens = getattr(response.usage, "input_tokens", None)
                output_tokens = getattr(response.usage, "output_tokens", None)
        except Exception:
            pass
        self.response_data = {"model": model, "input_tokens": input_tokens, "output_tokens": output_tokens}

        if not is_structured or response_format is None:
            return content

        # 结构化输出：解析 + pydantic 校验；失败则用 FixPrompt 修复后再校验
        try:
            return self._parse_structured(content, response_format)
        except Exception:
            print(f"Content is not valid JSON or schema-mismatched, trying to fix: {content}")
            fixed = self._reparse_with_fix_prompt(effective_system, content, model=model, temperature=temperature)
            return self._parse_structured(fixed, response_format)

class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini", "dashscope"] ="dashscope"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        elif self.provider == "dashscope":
            self.processor = BaseDashscopeProcessor()

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        history=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            history=history,
            **kwargs
        )

    # 多轮对话上下文最大条数，与 BaseDashscopeProcessor 一致
    HISTORY_MAX_MESSAGES = 8

    def get_answer_from_rag_context(self, question, rag_context, schema, model, history: Optional[List[Dict[str, str]]] = None, system_prompt_override: Optional[str] = None):
        """
        基于 RAG 上下文生成答案。支持多轮对话：传入 history，返回 (answer_dict, new_history)。
        history 仅保留最近 HISTORY_MAX_MESSAGES 条，超过部分直接删除。
        system_prompt_override: 若提供则替代按 schema 构建的 system 提示（用于无检索上下文时，JSON 格式仍由 schema 决定）。
        """
        history = history if history is not None else []
        history_trimmed = history[-self.HISTORY_MAX_MESSAGES:] if len(history) > self.HISTORY_MAX_MESSAGES else history

        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        if system_prompt_override is not None:
            system_prompt = system_prompt_override
        current_user_content = user_prompt.format(context=rag_context, question=question)

        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=current_user_content,
            is_structured=True,
            response_format=response_format,
            history=history_trimmed,
        )
        self.response_data = self.processor.response_data

        # 检查返回的字典是否包含所需字段，兼容 relevant_sources / relevant_pages
        if not isinstance(answer_dict, dict) or 'step_by_step_analysis' not in answer_dict:
            # 如果是dashscope返回的基本格式，尝试保留其内容
            if isinstance(answer_dict, dict) and 'final_answer' in answer_dict:
                # 这是dashscope处理后的格式，尝试从final_answer中提取结构化信息
                final_answer_content = answer_dict.get("final_answer", "N/A")
                
                # 如果final_answer是字符串且包含结构化信息，尝试解析
                if isinstance(final_answer_content, str) and final_answer_content.strip().startswith('{'):
                    try:
                        structured_data = json.loads(final_answer_content)
                        answer_dict = structured_data
                    except json.JSONDecodeError:
                        # 如果final_answer不是JSON，保持原有结构
                        answer_dict = {
                            "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                            "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                            "relevant_sources": answer_dict.get("relevant_sources", []),
                            "final_answer": answer_dict.get("final_answer", "N/A")
                        }
                else:
                    answer_dict = {
                        "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                        "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                        "relevant_sources": answer_dict.get("relevant_sources", []),
                        "final_answer": answer_dict.get("final_answer", "N/A")
                    }
            else:
                answer_dict = {
                    "step_by_step_analysis": "",
                    "reasoning_summary": "",
                    "relevant_sources": [],
                    "final_answer": "N/A"
                }
        # 本轮助理回复摘要（供下一轮上下文使用），在兼容性处理之后计算
        answer_summary = answer_dict.get("final_answer", "") if isinstance(answer_dict, dict) else str(answer_dict)
        if isinstance(answer_dict, dict) and not answer_summary:
            answer_summary = json.dumps(answer_dict, ensure_ascii=False)
        new_history = (history_trimmed + [
            # 记忆只保留历史对话：不要把检索到的上下文写入 history
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_summary},
        ])[-self.HISTORY_MAX_MESSAGES:]
        return answer_dict, new_history

    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = True if self.provider == "ibm" or self.provider == "gemini" else False
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        elif schema == "string":
            # 新增：支持开放性文本问题
            system_prompt = (prompts.AnswerWithRAGContextStringPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextStringPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextStringPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextStringPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict
