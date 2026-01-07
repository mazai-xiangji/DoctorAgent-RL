import json
import torch
import time
import os
import re
import argparse
# We will not use AutoModelForCausalLM and AutoTokenizer for the doctor model
# from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict

# Import OpenAI for DeepSeek-v3 API calls
from openai import OpenAI, AzureOpenAI
from openai import APIError, APIConnectionError, RateLimitError # Import specific exceptions

# Removed transformers imports as we are using API for all roles
# from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加分布式训练相关的全局变量
global_rank = None
global_world_size = None
global_local_rank = None

def setup_distributed():
    """Setup distributed training environment"""
    global global_rank, global_world_size, global_local_rank

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        # In a distributed setting, if CUDA is not available, we cannot proceed with nccl backend
        # For this script, we'll just return False and run in non-distributed mode
        return False

    # Get local rank from environment variable
    # LOCAL_RANK is typically set by the distributed launcher (e.g., torch.distributed.launch)
    global_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if global_local_rank == -1:
        # If LOCAL_RANK is not set, assume single GPU or CPU execution
        print("LOCAL_RANK environment variable not set. Using single GPU or CPU.")
        return False

    # Initialize process group
    # 'nccl' backend is recommended for GPU
    try:
        dist.init_process_group(backend="nccl")
    except Exception as e:
        print(f"Failed to initialize distributed process group: {e}")
        print("Falling back to non-distributed execution.")
        return False


    # Get rank and world size
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()

    # Set device for this process
    torch.cuda.set_device(global_local_rank)

    print(f"Initialized process {global_rank}/{global_world_size} on GPU {global_local_rank}")
    return True

class MedicalDialogueSimulation:
    def _create_client(self, model_name, base_url=None, api_key=None, role=None):
        """根据模型名称和角色创建 API 客户端"""
        # 尝试获取特定角色的 API KEY
        if not api_key and role:
            api_key = os.environ.get(f"{role.upper()}_API_KEY")
        
        # 如果没有特定角色的 KEY，使用通用的 API_KEY
        if not api_key:
            api_key = os.environ.get("API_KEY")
            
        if not api_key:
             print(f"Warning: No API key found for role {role} (model {model_name}). Please set {role.upper()}_API_KEY or API_KEY.")

        # 如果提供了 base_url，直接使用
        if base_url:
            return OpenAI(
                base_url=base_url,
                api_key=api_key,
            )

    def __init__(self, doctor_model_name, input_file, output_file, 
                 temperature=0.7, top_p=0.9, 
                 patient_model_name="deepseek-chat", assistant_model_name="deepseek-chat", 
                 batch_size=8,
                 doctor_base_url=None, doctor_api_key=None,
                 patient_base_url=None, patient_api_key=None,
                 assistant_base_url=None, assistant_api_key=None):
        """
        初始化医患对话模拟系统
        Args:
            doctor_model_name (str): 医生模型名称.
            input_file (str): 包含对话数据的输入 JSON 文件路径.
            output_file (str): 保存模拟结果的输出 JSON 文件路径.
            temperature (float): 医生模型生成时的温度参数.
            top_p (float): 医生模型生成时的 top-p 参数.
            patient_model_name (str): 患者模型名称 (API).
            assistant_model_name (str): 助理模型名称 (API).
            batch_size (int): 批处理大小 (用于并发API调用).
            doctor_base_url (str): 医生模型 API Base URL.
            doctor_api_key (str): 医生模型 API Key.
            patient_base_url (str): 患者模型 API Base URL.
            patient_api_key (str): 患者模型 API Key.
            assistant_base_url (str): 助理模型 API Base URL.
            assistant_api_key (str): 助理模型 API Key.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.doctor_model_name = doctor_model_name
        self.patient_model_name = patient_model_name
        self.assistant_model_name = assistant_model_name
        self.max_api_retries = 3

        # 设置分布式环境
        is_distributed = setup_distributed()
        if is_distributed:
            self.device = torch.device(f"cuda:{global_local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Process {global_rank if is_distributed else 0}: Using device: {self.device}")

        # 初始化 API 客户端
        self.doctor_client = self._create_client(self.doctor_model_name, base_url=doctor_base_url, api_key=doctor_api_key, role="doctor")
        self.patient_client = self._create_client(self.patient_model_name, base_url=patient_base_url, api_key=patient_api_key, role="patient")
        self.assistant_client = self._create_client(self.assistant_model_name, base_url=assistant_base_url, api_key=assistant_api_key, role="assistant")
        
        print(f"Initialized API clients for models: Doctor={self.doctor_model_name}, Patient={self.patient_model_name}, Assistant={self.assistant_model_name}")

        # 医生系统提示词
        self.doctor_system_prompt = """作为一名经验丰富的老中医，你的任务是对病人进行中医问诊，并最终给出诊断。

【核心任务】
1. **问诊**: 主动、多次提问以获取充足信息。每次只问一到两个核心问题，避免信息混乱。
2. **诊断**: 在信息充分后，及时给出明确的诊断结果（包括病名和中医证型）、诊断依据及治疗方案。
3. **效率**: 请力求在 **10轮对话** 内完成。优先询问关键问题（如主诉、二便、饮食、睡眠等），避免闲聊。

【输出规则（必须严格遵守）】
你必须先进行思考，输出 `<think>...</think>`，然后根据思考结果，选择以下**三种格式之一**作为你回复的内容：

1. **询问患者** (用于询问主观感受、过往病史、生活习惯)：
   格式示例：<对患者>您最近睡眠怎么样？

2. **询问助理** (用于查询客观信息，如舌象、脉象、检查报告)：
   格式示例：<对助理>请提供患者的舌象和脉象。

3. **提交诊断** (用于向专家汇报最终结果，包含诊断、依据和治疗方案)：
   格式示例：<对专家>诊断结果：肝郁脾虚证。诊断依据：... 治疗方案：...
请严格遵循此流程：先输出 <think> [你的思考] </think>，然后直接输出你的问诊或诊断内容。不要输出任何额外的标签、解释或说明。"""

        # 患者系统提示词
        self.patient_system_prompt = """你是一名患者。请根据你的身体状况（自述信息）回答医生的问题。
回答要简洁明了，只回答医生问的问题。如果医生问的问题你不知道或与你的病情无关，请如实回答。
不要主动提供诊断结果。"""

        # 助理系统提示词
        self.assistant_system_prompt = """你是一名医疗助理。你的任务是根据患者的详细病历信息（包括舌象、脉象、检查结果等客观信息）回答医生的询问。
请只提供客观事实，不要进行诊断。如果医生询问的信息在病历中找不到，请回答“病历中未记录相关信息”。"""

        # 最终诊断提示词
        self.final_diagnosis_system_prompt = """You are an experienced doctor who must provide a diagnosis and recommendation based on existing information. You have already asked enough questions and must now give the final diagnosis and treatment advice.

Based on the available information, please provide your best possible diagnosis and recommendation, even if the information is incomplete.

Respond strictly in the following format:
<answer>
Diagnosis: (the most likely disease or condition)
Recommendation: (corresponding treatment plan or suggestion)
</answer>

Do not include any other content. Be concise.
"""

    def load_dialogue_data(self):
        """加载对话数据"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_doctor_questions_and_patient_responses(self, dialogue):
        """从对话中提取医生问题和对应的患者回答"""
        questions_responses = []

        for turn in dialogue:
            doctor_questions = turn.get("doctor_question", [])
            patient_responses = turn.get("patient_response", [])

            if not isinstance(doctor_questions, list):
                doctor_questions = [doctor_questions]
            if not isinstance(patient_responses, list):
                patient_responses = [patient_responses]

            if doctor_questions and patient_responses:
                questions_responses.append({
                    "doctor_questions": doctor_questions,
                    "patient_responses": patient_responses
                })

        return questions_responses

    def process_doctor_response(self, doctor_response):
        """处理医生的回应，判断是继续提问还是给出诊断"""
        ori_doctor_response = doctor_response.strip()
        
        # 提取 <think> 内容 (可选)
        think_content = ""
        think_match = re.search(r"<think>(.*?)</think>", ori_doctor_response, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            # 移除 think 标签，只处理剩下的内容
            doctor_response = re.sub(r"<think>.*?</think>", "", ori_doctor_response, flags=re.DOTALL).strip()
        else:
            doctor_response = ori_doctor_response

        # 解析指令
        if "<对患者>" in doctor_response:
            content = doctor_response.split("<对患者>")[1].strip()
            # 移除可能的结束标签
            content = content.split("</对患者>")[0].strip()
            return {"type": "question_patient", "target": "patient", "content": content, "think": think_content}
        
        elif "<对助理>" in doctor_response:
            content = doctor_response.split("<对助理>")[1].strip()
            content = content.split("</对助理>")[0].strip()
            return {"type": "question_assistant", "target": "assistant", "content": content, "think": think_content}
            
        elif "<对专家>" in doctor_response:
            content = doctor_response.split("<对专家>")[1].strip()
            content = content.split("</对专家>")[0].strip()
            
            # 尝试提取诊断和建议
            diagnosis = ""
            advice = ""
            if "诊断结果：" in content:
                parts = content.split("诊断结果：")
                if len(parts) > 1:
                    diagnosis_part = parts[1]
                    if "诊断依据：" in diagnosis_part:
                        diagnosis = diagnosis_part.split("诊断依据：")[0].strip()
                    elif "治疗方案：" in diagnosis_part:
                        diagnosis = diagnosis_part.split("治疗方案：")[0].strip()
                    else:
                        diagnosis = diagnosis_part.strip()
            
            if "治疗方案：" in content:
                advice = content.split("治疗方案：")[1].strip()
            
            return {"type": "diagnosis", "target": "expert", "diagnosis": diagnosis, "advice": advice, "content": content, "think": think_content}
            
        else:
            # 默认情况，如果没有匹配到标签，假设是对患者提问
            return {"type": "question_patient", "target": "patient", "content": doctor_response, "think": think_content}

    def count_tokens(self, text):
        """计算文本的token数量 (简单估算)"""
        return len(text) // 4

    def generate_final_diagnosis(self, dialogue_history_messages):
        """
        生成最终诊断和建议 (用于达到最大轮次时强制诊断)
        Args:
            dialogue_history_messages (list): 对话历史消息列表.
        Returns:
            dict: 包含诊断和建议的结果.
        """
        messages = [
            {"role": "system", "content": self.final_diagnosis_system_prompt}
        ]

        messages.extend(dialogue_history_messages)
        messages.append({"role": "user", "content": "信息收集已完成，请根据上述对话提供最终诊断和建议"})

        response = "Error generating diagnosis after multiple retries."
        prompt_tokens = self.count_tokens(json.dumps(messages)) # Approximate token count on error
        response_tokens = 0

        for attempt in range(self.max_api_retries):
            try:
                # 调用 DeepSeek-v3 API 生成最终诊断
                completion = self.doctor_client.chat.completions.create(
                    model=self.doctor_model_name,
                    messages=messages,
                    # temperature=self.temperature,
                    # top_p=self.top_p,
                    max_tokens=1024 # Control generated token count
                )
                response = completion.choices[0].message.content.strip()
                prompt_tokens = completion.usage.prompt_tokens
                response_tokens = completion.usage.completion_tokens
                break # Break out of retry loop on success

            except (APIError, APIConnectionError, RateLimitError) as e:
                print(f"Attempt {attempt + 1}/{self.max_api_retries} failed for final diagnosis: {e}")
                if attempt < self.max_api_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    print(f"Max retries reached for final diagnosis.")
            except Exception as e:
                 print(f"An unexpected error occurred during final diagnosis attempt {attempt + 1}/{self.max_api_retries}: {e}")
                 if attempt < self.max_api_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                 else:
                    print(f"Max retries reached for final diagnosis.")


        # 解码响应并提取诊断和建议
        diagnosis_match = re.search(r"Diagnosis[:：](.*?)(?=Recommendation[:：]|$)", response, re.DOTALL)
        advice_match = re.search(r"Recommendation[:：](.*?)(?=\n|$)", response, re.DOTALL)

        diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else "Unable to determine diagnosis"
        advice = advice_match.group(1).strip() if advice_match else "Recommend further examination"

        return {
            "type": "diagnosis",
            "diagnosis": diagnosis,
            "advice": advice,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }

    def batch_generate_doctor_responses(self, dialogue_states):
        """
        批量生成医生的回复 (问题或诊断)。
        Args:
            dialogue_states (list): 需要生成医生回复的对话状态列表。
                                    每个状态包含: 'id', 'patient_context', 'dialogue_history_messages', 'iteration' 等。
        Returns:
            list: 包含每个对话医生回复结果的列表。
                  每个结果包含: 'dialogue_id', 'type', 'content'/'diagnosis'/'advice', 'tokens', 'prompt_tokens'。
        """
        if not dialogue_states:
            return []

        processed_results = []
        # Iterate through each dialogue state in the batch and call the API individually
        for state in dialogue_states:
            # 构建当前轮次的 prompt
            messages = [
                # {"role": "system", "content": self.doctor_system_prompt}, # System prompt is included in the first user message as per original logic
                {"role": "user", "content": self.doctor_system_prompt}
            ]
            # 添加对话历史
            messages.extend(state['dialogue_history_messages'])

            response = "Error generating response after multiple retries."
            prompt_tokens = self.count_tokens(json.dumps(messages)) # Approximate token count on error
            response_tokens = 0
            ori_response = response # Default original response in case of failure

            for attempt in range(self.max_api_retries):
                try:
                    completion = self.doctor_client.chat.completions.create(
                        model=self.doctor_model_name,
                        messages=messages,
                        # temperature=self.temperature,
                        # top_p=self.top_p,
                        max_tokens=1024 # Control generated token count
                    )
                    response = completion.choices[0].message.content.strip()
                    prompt_tokens = completion.usage.prompt_tokens
                    response_tokens = completion.usage.completion_tokens
                    ori_response = response # Update original response on success
                    break # Break out of retry loop on success

                except (APIError, APIConnectionError, RateLimitError) as e:
                    print(f"Attempt {attempt + 1}/{self.max_api_retries} failed for dialogue {state['id']}: {e}")
                    if attempt < self.max_api_retries - 1:
                        time.sleep(2 ** attempt) # Exponential backoff
                    else:
                        print(f"Max retries reached for dialogue {state['id']}.")
                except Exception as e:
                    print(f"An unexpected error occurred during dialogue {state['id']} attempt {attempt + 1}/{self.max_api_retries}: {e}")
                    if attempt < self.max_api_retries - 1:
                        time.sleep(2 ** attempt) # Exponential backoff
                    else:
                        print(f"Max retries reached for dialogue {state['id']}.")


            processed_response = self.process_doctor_response(response)
            # 添加 token 统计信息
            processed_response["prompt_tokens"] = prompt_tokens
            processed_response["tokens"] = response_tokens

            # 包含原始对话的状态信息，以便后续更新
            processed_response["dialogue_id"] = state['id']
            processed_response["ori_response"] = ori_response # Use updated ori_response

            processed_results.append(processed_response)

        return processed_results

    def batch_generate_api_responses(self, dialogue_states, role="patient"):
        """
        批量生成角色(患者/助理)的回复 (API版本)。
        Args:
            dialogue_states (list): 对话状态列表。
            role (str): "patient" or "assistant".
        Returns:
            list: 包含回复结果的列表。
        """
        if not dialogue_states:
            return []

        import concurrent.futures

        results = [None] * len(dialogue_states)

        def process_single_dialogue(index, state):
            try:
                # 构建 System Message 和 Context
                if role == "patient":
                    context = state.get('patient_context', '')
                    system_content = f"{self.patient_system_prompt}\n\nPatient Context:\n{context}"
                elif role == "assistant":
                    context = state.get('assistant_context', '')
                    system_content = f"{self.assistant_system_prompt}\n\nPatient Record:\n{context}"
                else:
                    return None

                messages = [{"role": "system", "content": system_content}]

                # 1. 添加初始问题 (总是存在的，且总是医生发起的)
                # 如果是助理，不需要初始问题（因为那是问患者的）
                if role != "assistant":
                    start_question = "你好，请问你哪里不舒服？"
                    messages.append({"role": "user", "content": start_question})

                # 2. 遍历 simulation_dialogue 构建历史
                sim_dialogue = state.get('simulation_dialogue', [])
                for turn in sim_dialogue:
                    turn_role = turn['role']
                    content = turn['content']
                    target = turn.get('target')
                    
                    if turn_role == "doctor":
                        # 如果是助理，只看针对助理的问题
                        if role == "assistant" and target != "assistant":
                            continue
                        messages.append({"role": "user", "content": content})
                    elif turn_role == role: # 如果是当前角色 (Patient or Assistant)
                        messages.append({"role": "assistant", "content": content})
                    else: # 是另一个角色 (Assistant or Patient)
                        # 如果当前是助理角色，且历史消息是患者发的，则跳过
                        if role == "assistant" and turn_role == "patient":
                            continue

                        # 作为 User 消息，但注明身份，以便区分
                        other_role_name = "Assistant" if turn_role == "assistant" else "Patient"
                        messages.append({"role": "user", "content": f"[{other_role_name}]: {content}"})

                client = self.patient_client if role == "patient" else self.assistant_client
                completion = client.chat.completions.create(
                    model=self.patient_model_name if role == "patient" else self.assistant_model_name,
                    messages=messages,
                    max_tokens=1024
                )
                response = completion.choices[0].message.content.strip()
                
                return {
                    "dialogue_id": state['id'],
                    "content": response,
                    "original_state_index": index
                }
            except Exception as e:
                print(f"Error generating {role} response: {e}")
                return {
                    "dialogue_id": state['id'],
                    "content": "Error generating response.",
                    "original_state_index": index
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {executor.submit(process_single_dialogue, i, state): i for i, state in enumerate(dialogue_states)}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results[result['original_state_index']] = result

        return results


    def run_simulation(self, max_iterations=10, start_idx=0, end_idx=None, batch_size=8):
        """
        运行医患对话模拟，支持批处理。
        Args:
            max_iterations (int): 最大对话轮次.
            start_idx (int): 开始处理的对话索引 (基于原始数据集).
            end_idx (int): 结束处理的对话索引 (不包含) (基于原始数据集).
            batch_size (int): 批处理大小 (主要影响患者模型).
        Returns:
            list: 所有对话的模拟结果列表.
        """
        self.batch_size = batch_size # 更新批处理大小

        dialogue_data = self.load_dialogue_data()

        # 根据 start_idx 和 end_idx 截取需要处理的数据
        if end_idx is None:
            end_idx = len(dialogue_data)
        # 存储原始索引和对应的数据
        dialogue_data_subset = [(i, dialogue_data[i]) for i in range(start_idx, end_idx)]

        # 在分布式环境中，每个进程只处理部分数据
        if global_world_size and global_world_size > 1:
            # 计算每个进程处理的数据量
            per_rank = len(dialogue_data_subset) // global_world_size
            remainder = len(dialogue_data_subset) % global_world_size

            # 计算当前进程的起始和结束索引
            start_idx = global_rank * per_rank + min(global_rank, remainder)
            end_idx = start_idx + per_rank + (1 if global_rank < remainder else 0)

            # 只处理分配给当前进程的数据
            dialogue_data_subset = dialogue_data_subset[start_idx:end_idx]
            print(f"Process {global_rank}: Processing {len(dialogue_data_subset)} items (indices {start_idx}-{end_idx-1})")

        # --- Resume Logic Start ---
        # Determine checkpoint file
        if global_world_size and global_world_size > 1:
            root, ext = os.path.splitext(self.output_file)
            checkpoint_file = f"{root}_rank{global_rank}{ext}"
        else:
            checkpoint_file = self.output_file

        simulation_results = []
        completed_ids = set()

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    simulation_results = json.load(f)
                completed_ids = set(item['id'] for item in simulation_results)
                print(f"Process {global_rank if global_rank is not None else 0}: Resuming from {checkpoint_file}, found {len(simulation_results)} completed dialogues.")
            except Exception as e:
                print(f"Process {global_rank if global_rank is not None else 0}: Warning: Could not load checkpoint {checkpoint_file}: {e}")
        
        def save_checkpoint():
            try:
                temp_file = checkpoint_file + ".tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(simulation_results, f, ensure_ascii=False, indent=2)
                os.replace(temp_file, checkpoint_file)
            except Exception as e:
                print(f"Process {global_rank if global_rank is not None else 0}: Error saving checkpoint: {e}")
        # --- Resume Logic End ---

        # 初始化活跃对话的状态列表
        active_dialogues = []
        for original_idx, item in dialogue_data_subset:
             # Check if already completed
             dialogue_id = item.get("id", original_idx)
             if dialogue_id in completed_ids:
                 continue

             patient_personal_info = item.get("患者个人信息", "")
             inquiry_info = item.get("问诊信息", "")
             other_info = item.get("其余信息", "")
             true_diagnosis = item.get("诊断结果", "")
             true_evidence = item.get("诊断依据", "")
             category = item.get("科室", "unknown")

             if not inquiry_info:
                 print(f"Dialogue {original_idx} is missing inquiry info. Skipping.")
                 continue

             patient_context = f"患者个人信息：{patient_personal_info}\n问诊信息：{inquiry_info}"
             assistant_context = f"问诊信息：{inquiry_info}\n其余信息：{other_info}"

             # Fixed start question
             start_question = "你好，请问你哪里不舒服？"

             # 初始化对话状态
             active_dialogues.append({
                 "id": item.get("id", original_idx), # 使用原数据字典的id作为对话 ID
                 "category": category,
                 "patient_context": patient_context,
                 "assistant_context": assistant_context,
                 "true_diagnosis": true_diagnosis,
                 "true_evidence": true_evidence,
                 "simulation_dialogue": [], # 存储模拟生成的对话轮次
                 "dialogue_history_messages": [{"role": "assistant", "content": start_question}],
                 "iteration": 0, # 当前对话轮次计数
                 "is_completed": False, # 对话是否已完成
                 "next_turn": "patient", # Initial turn is patient
                 "doctor_question": start_question # Store doctor's question
             })

        # simulation_results = [] # Already initialized above

        print(f"Starting simulation for {len(active_dialogues)} dialogues with batch size {self.batch_size}")

        # Main simulation loop, continue as long as there are active dialogues
        while active_dialogues:
            doctor_turn_dialogues = [] # List of dialogues needing a doctor response
            patient_turn_dialogues = [] # List of dialogues needing a patient response
            assistant_turn_dialogues = [] # List of dialogues needing an assistant response
            finished_dialogues_indices = [] # Indices of completed dialogues in active_dialogues

            # Categorize active dialogues into those needing a doctor, patient, or assistant response
            for i, state in enumerate(active_dialogues):
                if state['is_completed']:
                    finished_dialogues_indices.append(i)
                    continue # Skip completed dialogues

                # Check if maximum iterations reached
                if state['iteration'] >= max_iterations:
                     # Force final diagnosis
                     final_diagnosis = self.generate_final_diagnosis(state['dialogue_history_messages'])
                     diagnosis = final_diagnosis.get("diagnosis", "")
                     advice = final_diagnosis.get("advice", "")

                     # Record final diagnosis in simulation dialogue
                     state['simulation_dialogue'].append({
                         "turn": state['iteration'] + 1, # Diagnosis happens after the current iteration
                         "role": "doctor",
                         "content": f"Diagnosis: {diagnosis}\nRecommendation: {advice}",
                         "tokens": final_diagnosis.get("tokens", 0),
                         "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                         "is_diagnosis": True,
                         "is_forced": True # Mark as forced completion
                     })
                     state['is_completed'] = True # Mark as completed
                     finished_dialogues_indices.append(i) # Add to completed list
                     continue

                # Determine next turn
                if state['next_turn'] == "doctor":
                    doctor_turn_dialogues.append(state)
                elif state['next_turn'] == "patient":
                    patient_turn_dialogues.append(state)
                elif state['next_turn'] == "assistant":
                    assistant_turn_dialogues.append(state)
                else:
                    # Should not happen
                    print(f"Unknown turn type: {state['next_turn']}")
                    state['is_completed'] = True
                    finished_dialogues_indices.append(i)


            # Remove completed dialogues from active_dialogues and add to simulation_results
            # Remove from the end to avoid index issues
            for i in sorted(finished_dialogues_indices, reverse=True):
                 completed_dialogue = active_dialogues.pop(i)
                 simulation_results.append({
                     "id": completed_dialogue['id'],
                     "category": completed_dialogue.get('category', 'unknown'),
                     "patient_context": completed_dialogue['patient_context'],
                     "true_diagnosis": completed_dialogue['true_diagnosis'],
                     "true_evidence": completed_dialogue['true_evidence'],
                     "simulation_dialogue": completed_dialogue['simulation_dialogue'],
                     "total_turns": completed_dialogue['iteration'],
                     "is_completed": completed_dialogue['is_completed'],
                     "assistant_context": completed_dialogue.get('assistant_context', '')
                 })
            
            # Save checkpoint if any dialogues finished
            if finished_dialogues_indices:
                save_checkpoint()

            # Process dialogues needing a doctor response (using DeepSeek API - individual calls)
            if doctor_turn_dialogues:
                print(f"Processing {len(doctor_turn_dialogues)} doctor turns...")
                # Process doctor turns one by one as the API call is not inherently batched in this implementation
                for state in tqdm(doctor_turn_dialogues, desc="Doctor Turns", leave=False):
                    # Call the function to generate doctor response for this single state
                    # We pass a list containing only this state to reuse the batch_generate_doctor_responses structure
                    doctor_results = self.batch_generate_doctor_responses([state])

                    # Update the state based on the result (there will be only one result)
                    if doctor_results:
                        result = doctor_results[0]
                        # Find the corresponding state in active_dialogues (needed if the list order changes)
                        current_state = next((d for d in active_dialogues if d['id'] == result["dialogue_id"]), None)
                        if current_state:
                            current_state['iteration'] += 1 # Increment turn count after doctor's response

                            # Parse the doctor's response to determine the next action
                            target = result.get("target", "patient") # Default to patient if not specified
                            content = result.get("content", "")
                            thought = result.get("think", "")
                            
                            # Record doctor's response in simulation dialogue
                            current_state['simulation_dialogue'].append({
                                "turn": current_state['iteration'],
                                "role": "doctor",
                                "content": content,
                                "thought": thought,
                                "target": target,
                                "tokens": result.get("tokens", 0),
                                "prompt_tokens": result.get("prompt_tokens", 0)
                            })
                            
                            # Add doctor's response to dialogue history
                            current_state['dialogue_history_messages'].append({"role": "assistant", "content": result["ori_response"]})

                            if target == "patient":
                                current_state['next_turn'] = "patient"
                                current_state['doctor_question'] = content
                            elif target == "assistant":
                                current_state['next_turn'] = "assistant"
                                current_state['doctor_question'] = content
                            elif target == "expert":
                                # Diagnosis/Expert consultation - End of dialogue for this simulation scope
                                current_state['is_completed'] = True
                                current_state['simulation_dialogue'][-1]['is_diagnosis'] = True
                            else:
                                # Default or unknown, maybe end?
                                current_state['is_completed'] = True


            # Process dialogues needing a patient response (using API - batched)
            if patient_turn_dialogues:
                 print(f"Processing {len(patient_turn_dialogues)} patient turns in batches...")
                 # Process in batches
                 for i in tqdm(range(0, len(patient_turn_dialogues), self.batch_size), desc="Patient Batches", leave=False):
                     batch_states = patient_turn_dialogues[i:i + self.batch_size]
                     # Call the function to generate patient responses for this batch
                     patient_results = self.batch_generate_api_responses(batch_states, role="patient")

                     # Update each dialogue's state based on the results
                     for result in patient_results:
                         if result:
                             dialogue_id = result["dialogue_id"]
                             # Find the corresponding active dialogue state
                             state = next((d for d in active_dialogues if d['id'] == dialogue_id), None)
                             if state:
                                 patient_response_content = result["content"]
                                 # Record patient response in simulation dialogue
                                 state['simulation_dialogue'].append({
                                     "turn": state['iteration'], 
                                     "role": "patient",
                                     "content": patient_response_content
                                 })
                                 # Add patient response to dialogue history
                                 state['dialogue_history_messages'].append({"role": "user", "content": patient_response_content})
                                 # Next turn is doctor
                                 state['next_turn'] = "doctor"

            # Process dialogues needing an assistant response (using API - batched)
            if assistant_turn_dialogues:
                 print(f"Processing {len(assistant_turn_dialogues)} assistant turns in batches...")
                 # Process in batches
                 for i in tqdm(range(0, len(assistant_turn_dialogues), self.batch_size), desc="Assistant Batches", leave=False):
                     batch_states = assistant_turn_dialogues[i:i + self.batch_size]
                     # Call the function to generate assistant responses for this batch
                     assistant_results = self.batch_generate_api_responses(batch_states, role="assistant")

                     # Update each dialogue's state based on the results
                     for result in assistant_results:
                         if result:
                             dialogue_id = result["dialogue_id"]
                             # Find the corresponding active dialogue state
                             state = next((d for d in active_dialogues if d['id'] == dialogue_id), None)
                             if state:
                                 assistant_response_content = result["content"]
                                 # Record assistant response in simulation dialogue
                                 state['simulation_dialogue'].append({
                                     "turn": state['iteration'], 
                                     "role": "assistant",
                                     "content": assistant_response_content
                                 })
                                 # Add assistant response to dialogue history
                                 state['dialogue_history_messages'].append({"role": "user", "content": assistant_response_content})
                                 # Next turn is doctor
                                 state['next_turn'] = "doctor"


            # After processing all batches, check again for newly completed dialogues
            newly_finished_dialogues_indices = []
            for i, state in enumerate(active_dialogues):
                 if state['is_completed']:
                     newly_finished_dialogues_indices.append(i)

            # Remove newly completed dialogues
            for i in sorted(newly_finished_dialogues_indices, reverse=True):
                 completed_dialogue = active_dialogues.pop(i)
                 simulation_results.append({
                     "id": completed_dialogue['id'],
                     "category": completed_dialogue.get('category', 'unknown'),
                     "patient_context": completed_dialogue['patient_context'],
                     "true_diagnosis": completed_dialogue['true_diagnosis'],
                     "true_evidence": completed_dialogue['true_evidence'],
                     "simulation_dialogue": completed_dialogue['simulation_dialogue'],
                     "total_turns": completed_dialogue['iteration'],
                     "is_completed": completed_dialogue['is_completed'],
                     "assistant_context": completed_dialogue.get('assistant_context', '')
                 })
            
            # Save checkpoint if any dialogues finished
            if newly_finished_dialogues_indices:
                save_checkpoint()

            # If no progress was made in this iteration (no doctor or patient turns processed)
            # and there are still active dialogues, break the loop and force final diagnosis
            if not doctor_turn_dialogues and not patient_turn_dialogues and not assistant_turn_dialogues and active_dialogues:
                 print("Warning: No progress made in this iteration, breaking loop and forcing final diagnosis for remaining dialogues.")
                 # Force final diagnosis for remaining active dialogues
                 for state in active_dialogues:
                      if not state['is_completed']:
                           final_diagnosis = self.generate_final_diagnosis(state['dialogue_history_messages'])
                           diagnosis = final_diagnosis.get("diagnosis", "")
                           advice = final_diagnosis.get("advice", "")

                           state['simulation_dialogue'].append({
                               "turn": state['iteration'] + 1,
                               "role": "doctor",
                               "content": f"Diagnosis: {diagnosis}\nRecommendation: {advice}",
                               "tokens": final_diagnosis.get("tokens", 0),
                               "prompt_tokens": final_diagnosis.get("prompt_tokens", 0),
                               "is_diagnosis": True,
                               "is_forced": True
                           })
                           state['is_completed'] = True
                           simulation_results.append({
                               "id": state['id'],
                               "category": state.get('category', 'unknown'),
                               "patient_context": state['patient_context'],
                               "true_diagnosis": state['true_diagnosis'],
                               "true_evidence": state['true_evidence'],
                               "simulation_dialogue": state['simulation_dialogue'],
                               "total_turns": state['iteration'],
                               "is_completed": state['is_completed'],
                               "assistant_context": state.get('assistant_context', '')
                           })
                 active_dialogues = [] # Clear the list of active dialogues
                 save_checkpoint() # Save final state


        # In a distributed environment, collect results from all processes
        if global_world_size and global_world_size > 1:
            # Gather all results
            all_results = [None] * global_world_size
            dist.all_gather_object(all_results, simulation_results)

            # Only on the main process (rank 0), merge the results
            if global_rank == 0:
                # Flatten the list of results from all processes
                simulation_results = [result for process_results in all_results for result in process_results]
            else:
                # Non-main processes return an empty list
                simulation_results = []

        # Save the final results to a file
        if global_rank is None or global_rank == 0: # Only save from the main process or in non-distributed mode
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(simulation_results, f, ensure_ascii=False, indent=2)

            print(f"Simulation complete. Results saved to: {self.output_file}")

        return simulation_results


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Medical Dialogue Simulation with DeepSeek-v3 API (Doctor) and API (Patient/Assistant)")
    parser.add_argument("--doctor_model_name", type=str, default="deepseek-v3-250324",
                        help="DeepSeek-v3 model name for the doctor")
    parser.add_argument("--patient_model_name", type=str, default="gpt-4o",
                        help="Model name for the patient (API)")
    parser.add_argument("--assistant_model_name", type=str, default="gpt-4o",
                        help="Model name for the assistant (API)")
    parser.add_argument("--input_file", type=str, default="ragen/env/medical_consultation/evaluation/test.json",
                        help="Input JSON file containing dialogue data")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/deepseek-v3/", 
                        help="Output directory for the simulation results")
    parser.add_argument("--output_prefix", type=str, default="test_deepseek_en_output", 
                        help="Prefix for the output filename")
    parser.add_argument("--add_timestamp", action="store_true",
                        help="Add timestamp to output filename")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of dialogue turns")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for doctor model generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter for doctor model")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index for dialogue processing")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Ending index for dialogue processing (None means process to the end)")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for API calls")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    # API Configuration arguments
    parser.add_argument("--doctor_base_url", type=str, default=None, help="Base URL for doctor model API")
    parser.add_argument("--doctor_api_key", type=str, default=None, help="API Key for doctor model")
    parser.add_argument("--patient_base_url", type=str, default=None, help="Base URL for patient model API")
    parser.add_argument("--patient_api_key", type=str, default=None, help="API Key for patient model")
    parser.add_argument("--assistant_base_url", type=str, default=None, help="Base URL for assistant model API")
    parser.add_argument("--assistant_api_key", type=str, default=None, help="API Key for assistant model")


    # Add usage example
    parser.epilog = """
    Example usage:
    # Run simulation with DeepSeek-v3 as doctor and GPT-4o as patient/assistant
    # Ensure DEEPSEEK_API_KEY and OPENAI_API_KEY environment variables are set
    python your_script_name.py --doctor_model_name deepseek-v3-250324 --patient_model_name gpt-4o --assistant_model_name gpt-4o --input_file input.json --output_dir ./results --add_timestamp --batch_size 16
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    args = parser.parse_args()

    if args.verbose:
        print(f"Starting Medical Dialogue Simulation with API Models")
        print(f"Doctor model: {args.doctor_model_name}")
        print(f"Patient model: {args.patient_model_name}")
        print(f"Assistant model: {args.assistant_model_name}")
        print(f"Input file: {args.input_file}")
        print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
        print(f"Maximum iterations: {args.max_iterations}")
        print(f"Batch size: {args.batch_size}")
        if args.start_idx > 0 or args.end_idx is not None:
            print(f"Processing dialogues from index {args.start_idx} to {args.end_idx if args.end_idx is not None else 'end'}")

    # Create output filename
    output_filename = args.output_prefix

    # Add index range to filename if specified
    if args.start_idx > 0 or args.end_idx is not None:
        range_info = f"_{args.start_idx}_to_{args.end_idx if args.end_idx is not None else 'end'}"
        output_filename += range_info

    # Add timestamp if needed
    if args.add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename += f"_{timestamp}"

    output_filename += ".json"

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, output_filename)

    if args.verbose:
        print(f"Output will be saved to: {output_file}")
        print("Initializing models and API client...")

    start_time = time.time()

    # Create and run simulator
    simulator = MedicalDialogueSimulation(
        doctor_model_name=args.doctor_model_name,
        patient_model_name=args.patient_model_name,
        assistant_model_name=args.assistant_model_name,
        input_file=args.input_file,
        output_file=output_file,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        doctor_base_url=args.doctor_base_url,
        doctor_api_key=args.doctor_api_key,
        patient_base_url=args.patient_base_url,
        patient_api_key=args.patient_api_key,
        assistant_base_url=args.assistant_base_url,
        assistant_api_key=args.assistant_api_key
    )

    results = simulator.run_simulation(
        max_iterations=args.max_iterations,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size
    )

    # Calculate and report statistics
    end_time = time.time()
    total_time = end_time - start_time

    # Only report statistics from the main process or in non-distributed mode
    if global_rank is None or global_rank == 0:
        if args.verbose:
            print(f"\nSimulation complete in {total_time:.2f} seconds")
            print(f"Results saved to: {output_file}")

            # Calculate some basic statistics
            if results:
                total_dialogues = len(results)
                # Filter out results that might be None if a process failed or returned empty
                valid_results = [r for r in results if r is not None]
                completed_dialogues = sum(1 for r in valid_results if r.get('is_completed', False))
                # Calculate average turns from valid results
                avg_turns = sum(r.get('total_turns', 0) for r in valid_results) / len(valid_results) if valid_results else 0

                print(f"\nStatistics:")
                print(f"- Total dialogues processed (across all ranks): {total_dialogues}")
                print(f"- Completed dialogues: {completed_dialogues} ({completed_dialogues/total_dialogues*100:.1f}%)")
                print(f"- Average turns per dialogue: {avg_turns:.2f}")
            else:
                 print("\nNo results to report statistics.")


    # In a distributed environment, ensure all processes finish before exiting
    if global_world_size and global_world_size > 1:
        dist.barrier()
        if global_rank == 0:
             print(f"All processes completed.")
        # Clean up distributed environment
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
