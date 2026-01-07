#!/bin/bash

# ================= 配置区域 =================

# 1. 医生模型配置 (Doctor)
DOCTOR_MODEL="deepseek-v3-250324"
DOCTOR_API_KEY="sk-..."
DOCTOR_BASE_URL="https://api.deepseek.com/v1"

# 2. 患者模型配置 (Patient)
PATIENT_MODEL="gpt-4o"
PATIENT_API_KEY="sk-..."
PATIENT_BASE_URL="https://api.openai.com/v1"

# 3. 助理模型配置 (Assistant)
ASSISTANT_MODEL="gpt-4o"
ASSISTANT_API_KEY="sk-..."
ASSISTANT_BASE_URL="https://api.openai.com/v1"

# 4. 评估模型配置 (Evaluator - 用于语义评分)
EVAL_MODEL="gpt-4o"
EVAL_API_KEY="sk-..."
EVAL_BASE_URL="https://api.openai.com/v1"

# ===========================================

# 脚本路径
INFERENCE_SCRIPT="ragen/env/medical_consultation/evaluation/TCM_inference_fast_for_patientllm_with_api.py"
EVAL_SCRIPT="ragen/env/medical_consultation/evaluation/TCM_evaluation_for_patientllm_category.py"

# 数据路径
INPUT_DATA="Data/TCM/TCM_test.json"
OUTPUT_DIR="outputs/tcm_experiment"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 生成带时间戳的文件名前缀
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
INFERENCE_PREFIX="tcm_inference_${TIMESTAMP}"
INFERENCE_OUTPUT_FILE="${OUTPUT_DIR}/${INFERENCE_PREFIX}.json"
EVAL_OUTPUT_FILE="${OUTPUT_DIR}/tcm_eval_result_${TIMESTAMP}.json"

echo "========================================================"
echo "开始 TCM 评估流程"
echo "输入数据: $INPUT_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "医生模型: $DOCTOR_MODEL"
echo "========================================================"

# 1. 执行推理 (Inference)
echo "[1/2] 正在执行推理..."
python "$INFERENCE_SCRIPT" \
    --doctor_model_name "$DOCTOR_MODEL" \
    --doctor_base_url "$DOCTOR_BASE_URL" \
    --doctor_api_key "$DOCTOR_API_KEY" \
    --patient_model_name "$PATIENT_MODEL" \
    --patient_base_url "$PATIENT_BASE_URL" \
    --patient_api_key "$PATIENT_API_KEY" \
    --assistant_model_name "$ASSISTANT_MODEL" \
    --assistant_base_url "$ASSISTANT_BASE_URL" \
    --assistant_api_key "$ASSISTANT_API_KEY" \
    --input_file "$INPUT_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --output_prefix "$INFERENCE_PREFIX" \
    --max_iterations 10 \
    --batch_size 8 \
    --verbose

# 检查推理是否成功生成了文件
if [ ! -f "$INFERENCE_OUTPUT_FILE" ]; then
    echo "错误: 推理输出文件未找到: $INFERENCE_OUTPUT_FILE"
    exit 1
fi

echo "推理完成。结果已保存至: $INFERENCE_OUTPUT_FILE"

# 2. 执行评估 (Evaluation)
echo "========================================================"
echo "[2/2] 正在执行评估..."
echo "评估输入文件: $INFERENCE_OUTPUT_FILE"
echo "========================================================"

# 设置评估脚本所需的环境变量
export OPENAI_API_KEY="$EVAL_API_KEY"
export OPENAI_BASE_URL="$EVAL_BASE_URL"
export OPENAI_MODEL="$EVAL_MODEL"

python "$EVAL_SCRIPT" \
    --input_file "$INFERENCE_OUTPUT_FILE" \
    --output "$EVAL_OUTPUT_FILE" \
    --batch_size 16

# 检查评估是否成功
if [ -f "$EVAL_OUTPUT_FILE" ]; then
    echo "评估完成。最终结果已保存至: $EVAL_OUTPUT_FILE"
    echo "流程结束。"
else
    echo "错误: 评估结果文件未生成。"
    exit 1
fi
