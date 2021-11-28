pip install -r requirements.txt  &> /dev/null

pip install -q kaggle
apt-get install jq

mkdir ~/.kaggle
echo '{"username":"yunogasa1","key":"801b296373c90444e6c6f30f3fdb1933"}' | jq . > ~/.kaggle/kaggle.json

chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d sahilbannoo/yuno-search-data
kaggle datasets download -d sahilbannoo/yunosearchinfo

unzip yuno-search-data.zip && rm yuno-search-data.zip
unzip yunosearchinfo.zip && rm yuno-search-data.zip



