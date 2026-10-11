#!/usr/bin/env bash
set -e
python evaluate.py --split spider_dev --tag base
python evaluate.py --split spider_dev --tag dpo --adapter outputs/dpo-1.5b-s0/checkpoint-100
python evaluate.py --split spider_dev --tag grpo --adapter outputs/grpo-1.5b-s0/checkpoint-400
python evaluate.py --split spider_dev --tag rloo --adapter outputs/rloo-1.5b-s0/checkpoint-600
python evaluate.py --split spider_syn --tag base
python evaluate.py --split spider_syn --tag dpo --adapter outputs/dpo-1.5b-s0/checkpoint-100
python evaluate.py --split spider_syn --tag grpo --adapter outputs/grpo-1.5b-s0/checkpoint-400
python evaluate.py --split spider_syn --tag rloo --adapter outputs/rloo-1.5b-s0/checkpoint-600
python evaluate.py --split spider_dk --tag base
python evaluate.py --split spider_dk --tag dpo --adapter outputs/dpo-1.5b-s0/checkpoint-100
python evaluate.py --split spider_dk --tag grpo --adapter outputs/grpo-1.5b-s0/checkpoint-400
python evaluate.py --split spider_dk --tag rloo --adapter outputs/rloo-1.5b-s0/checkpoint-600
python evaluate.py --split bird_dev --tag base
python evaluate.py --split bird_dev --tag dpo --adapter outputs/dpo-1.5b-s0/checkpoint-100
python evaluate.py --split bird_dev --tag grpo --adapter outputs/grpo-1.5b-s0/checkpoint-400
python evaluate.py --split bird_dev --tag rloo --adapter outputs/rloo-1.5b-s0/checkpoint-600
