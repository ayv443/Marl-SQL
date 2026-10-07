#!/usr/bin/env bash
# bash download_data.sh
set -e
cd "$(dirname "$0")"
mkdir -p data eval_repos
pip install -q gdown

# Spider
if [ ! -d data/spider_data ]; then
  gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O data/spider.zip
  unzip -q data/spider.zip -d data && rm data/spider.zip
fi

# test suite databases
if [ ! -d data/testsuite_databases ]; then
  gdown 1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w -O data/testsuite.zip
  unzip -q data/testsuite.zip -d data && rm data/testsuite.zip
  [ -d data/database ] && mv data/database data/testsuite_databases
fi

# BIRD dev
if [ ! -d data/bird_dev ]; then
  curl -L -o data/bird_dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
  unzip -q data/bird_dev.zip -d data/bird_tmp && rm data/bird_dev.zip
  mv data/bird_tmp/dev_* data/bird_dev && rm -rf data/bird_tmp
  (cd data/bird_dev && unzip -q dev_databases.zip && rm -f dev_databases.zip)
  rm -rf data/bird_dev/__MACOSX
fi

# Spider-Syn / Spider-DK / Spider-Realistic
mkdir -p data/spider_variants
[ -d eval_repos/Spider-Syn ] || git clone -q https://github.com/ygan/Spider-Syn eval_repos/Spider-Syn
[ -d eval_repos/Spider-DK ]  || git clone -q https://github.com/ygan/Spider-DK  eval_repos/Spider-DK
cp eval_repos/Spider-Syn/Spider-Syn/dev.json data/spider_variants/spider_syn.json 2>/dev/null || echo "!! find Spider-Syn dev.json manually"
cp eval_repos/Spider-DK/Spider-DK.json      data/spider_variants/spider_dk.json  2>/dev/null || echo "!! find Spider-DK.json manually"
# Spider-Realistic: download spider-realistic.json from https://zenodo.org/record/5205322
# and save it as data/spider_variants/spider_realistic.json

# official eval scripts
[ -d eval_repos/test-suite-sql-eval ] || git clone -q https://github.com/taoyds/test-suite-sql-eval eval_repos/test-suite-sql-eval
[ -d eval_repos/mini_dev ] || git clone -q https://github.com/bird-bench/mini_dev eval_repos/mini_dev

echo "done"
ls data
