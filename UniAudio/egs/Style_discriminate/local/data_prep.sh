src1=$1
src2=$2
src3=$3
dst="./data"

mkdir -p $dst || exit 1

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1

python local/data_prepare.py $src1 $src2 $src3 $dst
