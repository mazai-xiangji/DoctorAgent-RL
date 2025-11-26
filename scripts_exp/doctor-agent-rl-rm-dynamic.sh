#!/bin/bash
# CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 5000 --tensor-parallel-size 1 --model /home/mdisk1/twx/qwen3base  --served-model-name API  --trust-remote-code --max-model-len 4096

WANDB_API_KEY="YOUR_API_KEY"

wandb login $WANDB_API_KEY

export RAY_DEBUG_POST_MORTEM=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES="2"

clip_ratio_low=0.2
clip_ratio_high=0.28
entropy_coeff=0.001

model_path=/home/mdisk2/ljx_new/Model/Qwen3-0.6B
exp_name=exp-medical-consultation-1ksft-patientllm-rm-random_t
project_name=RAGEN

#openai api 设置
base_url=""
api_key=""
model_name_api=""

python -m ragen.trainer.main_ppo \
  hydra.run.dir=outputs/exp_configs/logs/$(date +%Y-%m-%d)/$(date +%H-%M-%S) \
  data.train_files=data/MTMedDialog_RL.parquet \
  data.val_files=data/MTMedDialog_RL.parquet \
  data.train_data_num=null \
  data.val_data_num=64 \
  data.train_batch_size=1 \
  data.val_batch_size=1\
  data.max_prompt_length=1024 \
  data.max_response_length=256 \
  data.max_start_length=512 \
  data.max_obs_length=256 \
  data.shuffle=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=${model_path} \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  critic.ppo_micro_batch_size=1 \
  critic.optim.lr=1e-5 \
  critic.model.path=${model_path} \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_ctrl.kl_coef=0.001 \
  +algorithm.no_ref_policy=False \
  +actor_rollout_ref.actor.use_ref_policy=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
  +algorithm.no_think_rl=False \
  +algorithm.reward_norm_type=grpo \
  +actor_rollout_ref.actor.optim.betas=[0.9,0.95] \
  +critic.optim.betas=[0.9,0.95] \
  actor_rollout_ref.rollout.n_agent=8 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.actor.state_masking=True \
  trainer.logger=['wandb'] \
  +trainer.val_only=false \
  trainer.val_before_train=true \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=12 \
  trainer.test_freq=4 \
  trainer.project_name=${project_name} \
  trainer.experiment_name=${exp_name} \
  trainer.total_epochs=1 \
  trainer.total_training_steps=null \
  +trainer.ref_update_steps=null \
  env.name=medical_consultation_patient_llm_rm \
  env.use_env_llm=True \
  +env.max_turns=-1 \
  env.api_config.base_url=${base_url} \
  env.api_config.api_key=${api_key} \
  env.api_config.model=${model_name_api} \
  env.api_config.concurrency=4 \
  env.env_llm.fsdp_config.fsdp_size=-1 \
  env.env_llm.fsdp_config.param_offload=False \
  env.env_llm.vllm_config.tensor_parallel_size=4 \
  env.env_llm.vllm_config.gpu_memory_utilization=0.45 \
  env.env_llm.vllm_config.max_num_batched_tokens=8192 \
  env.env_llm.vllm_config.max_num_seqs=512 \
  env.env_llm.model.path=/home/mdisk2/ljx_new/Model/Qwen3-0.6B \
  env.env_llm.model.trust_remote_code=True \
  env.env_llm.model.use_liger=True \
  env.env_llm.model.override_config.max_position_embeddings=4000 \
  env.env_llm.generation.prompt_length=2272 \
  env.env_llm.generation.response_length=256 \
  env.env_llm.generation.max_model_len=2528 \
  env.env_llm.generation.temperature=0.01 \
  env.env_llm.generation.top_p=1.0 \
  env.env_llm.generation.top_k=-1 \
  env.env_llm.generation.repetition_penalty=1.0 \
  env.env_llm.generation.do_sample=False \
  env.env_llm.generation.num_beams=1 \
  env.env_llm.generation.best_of=1 \
  env.env_llm.generation.min_p=0.0 \
  env.env_llm.generation.n=1 \
  env.env_llm.generation.use_cache=True \
  env.env_llm.generation.use_beam_search=False \
  env.env_llm.generation.detokenize=False \
  env.env_llm.generation.ignore_eos=False \
  env.env_llm.generation.free_cache_engine=True \
  env.env_llm.generation.prompt_logprobs=0 \
  env.env_llm.generation.generation_logprobs=1 \
  env.env_llm.generation.disable_log_stats=True \
  env.env_llm.generation.dtype=bfloat16 \
  env.env_llm.generation.enforce_eager=True \
  env.env_llm.generation.enable_chunked_prefill=True \
  env.env_llm.generation.tensor_model_parallel_size=4 \
  env.env_llm.generation.gpu_memory_utilization=0.3 \
  env.env_llm.generation.max_tokens_per_batch=8192 \
  env.env_llm.generation.load_format=dummy_dtensor \
  env.env_llm.ulysses_sequence_parallel_size=1 \
  max_turns=10 \
  logging.log_images=true \
  logging.log_image_dir=log/trajectory \
  logging.log_image_step_size=4 \
  logging.log_n_image_per_batch=32 \
  2>&1

# target_model=DoctorLLM-7B-8k-GRPO-Clip_Higher-1000SFTCold-RM-DynamicTurn
# base_model=$model_path
# max_step=$(head -n 1 checkpoints/${project_name}/${exp_name}/latest_checkpointed_iteration.txt)
# ckpt_path=checkpoints/${project_name}/${exp_name}/global_step_${max_step}/actor

# bash merger.sh $base_model $ckpt_path $target_model

# bash eval_medical_patientllm_category.sh $target_model
