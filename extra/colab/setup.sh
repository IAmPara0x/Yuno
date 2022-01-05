# This script is used to download dataset from kaggle to colab.
set -e

pip install -r yuno/requirements.txt  &> /dev/null

pip install -q kaggle
apt-get install gnupg

mkdir ~/.kaggle
gpg --batch --output ~/.kaggle/kaggle.json --passphrase yunogasai --decrypt yuno/colab/key.json.gpg

chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d sahilbannoo/yuno-search-data
kaggle datasets download -d sahilbannoo/yunosearchinfo

unzip yuno-search-data.zip && rm yuno-search-data.zip
unzip yunosearchinfo.zip && rm yunosearchinfo.zip

