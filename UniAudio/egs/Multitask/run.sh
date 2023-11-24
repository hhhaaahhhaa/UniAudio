
# A demo recipe for VC based on LibriTTS dataset.
# VC: smantic token + prompt ---> wave
. ./path.sh

# pip3 install fairseq==0.12.2 einops==0.6.0 sentencepiece encodec

stage=4
stop_stage=100
ngpu=1  # how many GPUs, you want to use to train the model

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
data_tag='Multi'

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

### Stage 4: Training ###
if [ -z $data_tag ] && [ $stop_stage -le 4 ]; then
    echo "you should provide data tag" || exit 1;
fi

train_data_jsons="../TVC/data/train/${ngpu}splits/data_tvc.ALL.json ../TTS/data/train-clean-100/${ngpu}splits/data_tts.ALL.json ../VC/data/train-clean-100/${ngpu}splits/data_vc.ALL.json"
valid_data_jsons="../TVC/data/validation/${ngpu}splits/data_tvc.ALL.json ../TTS/data/dev-clean/${ngpu}splits/data_tts.ALL.json ../VC/data/dev-clean/${ngpu}splits/data_vc.ALL.json"

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
        --n_layer 6 \
        --n_head 16 \
        --n_embd 1536 \
        $train_opts
fi
