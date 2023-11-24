
# A demo recipe for VC based on LibriTTS dataset.
# VC: smantic token + prompt ---> wave
. ./path.sh

# pip3 install fairseq==0.12.2 einops==0.6.0 sentencepiece encodec

stage=1
stop_stage=100
ngpu=1  # how many GPUs, you want to use to train the model

train_set="train"
valid_set="validation"
test_sets="test"

# training config
seed=999
debug=true
batch_scale=4000 # the total number of tokens in one batch
learning_rate=0.005 # the learning rate
port=12345
train_opts=
inference_opts=
tag=
inference_tag=default
resume=
data_tag='style_disc'
TASK='style_disc'

if [ ! -d "utils" ]; then
  ln -s ../tools/kaldi/utils ./
fi
if [ ! -d "data_scripts" ]; then
  ln -s ../tools/data_scripts ./
fi

. utils/parse_options.sh

if [ ! -z $resume ]; then
    train_opts="--resume $resume"
    inference_opts="--resume $resume"
fi

if [ $debug == true ]; then
    export HOST_GPU_NUM=1
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"

else
    export HOST_GPU_NUM=8
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"
fi

### stage 1-3: data preparation ###

# Prepare data following Espnet and split
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare dataset"
    # this part aims to get the information about the dataset. 
    # Considering different tasks using different dataset, we donot provide the scripts to access dataset
    # for audio data, please prepare wav.scp and source_wav.scp
    # for prompt, please prepare instruction.scp
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the data for $ngpu GPUs"

    for part in $test_sets $valid_set $train_set; do
      mkdir -p data/${part}/${ngpu}splits
      # extra shuf to ensure balance across GPUs
      # So the generated data cannot be reproduced due to the shuffle randomness
      cat data/${part}/wav.scp | shuf >  data/${part}/wav.scp.shuf
      split_scp=
      for n in `seq 1 $ngpu`; do
          split_scp="$split_scp data/${part}/${ngpu}splits/wav.${n}.scp"
      done
      utils/split_scp.pl data/${part}/wav.scp.shuf $split_scp

    done
fi

# stage 2-3 process sequences respectively
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Prepare text and audio sequence"
    for part in $valid_set $train_set; do
    # for part in $valid_set; do
      echo "prepare $part ... "

      # split source_wav.scp based on wav.scp (target)
      utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/filter_source_wav.JOB.log \
        python3 data_scripts/filter_scp.py \
          data/${part}/${ngpu}splits/wav.JOB.scp data/${part}/source_wav.scp \
          data/${part}/${ngpu}splits/source_wav.JOB.scp || exit 1;

      # Instruction
      utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/filter_instruction.JOB.log \
        python3 data_scripts/filter_scp.py \
          data/${part}/${ngpu}splits/wav.JOB.scp data/${part}/instruction.scp \
          data/${part}/${ngpu}splits/instruction.JOB || exit 1;

      # Label
      utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/filter_label.JOB.log \
        python3 data_scripts/filter_scp.py \
          data/${part}/${ngpu}splits/wav.JOB.scp data/${part}/label.scp \
          data/${part}/${ngpu}splits/label.JOB || exit 1;
      
      # Audio Source
      utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/audio_source_codec_dump.JOB.log \
        python3 data_scripts/offline_tokenization.py \
          --input-file data/${part}/${ngpu}splits/source_wav.JOB.scp \
          --output-file data/${part}/${ngpu}splits/audio_source_codec.JOB.pt \
          --tokenizer audio --rank JOB || exit 1;

      # Audio Target
      utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/audio_codec_dump.JOB.log \
        python3 data_scripts/offline_tokenization.py \
          --input-file data/${part}/${ngpu}splits/wav.JOB.scp \
          --output-file data/${part}/${ngpu}splits/audio_codec.JOB.pt \
          --tokenizer audio --rank JOB || exit 1;
      
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "create data json"
    # json index from 0 but all data files index from 1
    for part in $valid_set $train_set; do
      for n in `seq 0 $[$ngpu-1]`; do
        python3 data_scripts/create_data_json.py \
         --task style_disc \
         --out-json   $PWD/data/${part}/${ngpu}splits/data_style_disc.${n}.json \
         --text_t5_seq $PWD/data/${part}/${ngpu}splits/instruction.$[$n+1] \
         --audio_source_seq $PWD/data/${part}/${ngpu}splits/audio_source_codec.$[$n+1].pt \
         --audio_seq  $PWD/data/${part}/${ngpu}splits/audio_codec.$[$n+1].pt \
         --label $PWD/data/${part}/${ngpu}splits/label.$[$n+1] \
         & 
      done; wait

    done
fi

### Stage 4: Training ###
if [ -z $data_tag ] && [ $stop_stage -le 4 ]; then
    echo "you should provide data tag" || exit 1;
fi

train_data_jsons="data/${train_set}/${ngpu}splits/data_style_disc.ALL.json"
valid_data_jsons="data/${valid_set}/${ngpu}splits/data_style_disc.ALL.json"

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    mkdir -p exp 
    if [ -z $tag ]; then
        echo "please provide a tag for this experiment" && exit 1;
    fi
    echo "stage 5: training..."
    NCCL_DEBUG=TRACE torchrun \
        --nproc_per_node ${HOST_GPU_NUM} --master_port $port \
        --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
        ../../train.py \
        --exp_dir exp \
        --seed $seed \
        --cudnn_deterministic \
        --train_data_jsons $train_data_jsons \
        --valid_data_jsons $valid_data_jsons \
        --batch_scale $batch_scale \
        --learning_rate $learning_rate \
        --non-acoustic-repeat 3 \
        --audio-tokenizer "soundstream" \
        --audio-prompt-tokenizer "audio_prompt" \
        --phone-tokenizer "alignment" \
        --semantic-tokenizer "hubert" \
        --semantic-tokenizer-duplicate true \
        --singPhoneTokenizer "sing_phone" \
        --singMidiTokenizer "sing_midi" \
        --FrozenT5Embedder "text_t5" \
        --sv-bool-tokenizer "sv_bool" \
        --n_layer 6 \
        --n_head 16 \
        --n_embd 1536 \
        $train_opts
fi
