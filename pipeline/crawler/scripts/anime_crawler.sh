
for ARGUMENT in "$@"
  do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
      CREATE_PROXY) CREATE_PROXY=${VALUE};;
      START) START=${VALUE} ;;
      END) END=${VALUE} ;;
      MONGODB_URL) MONGODB_URL=${VALUE} ;;
      USE_CACHED_PROXY) USE_CACHED_PROXY=${VALUE} ;;
      *)
    esac
  done


if [ "$CREATE_PROXY" == "True" ];
then
timeout --signal=SIGINT 30 proxybroker find --types HTTPS > proxy-list.txt
./preprocess.py proxy-list.txt
fi

scrapy crawl anime-crawler -a start="$START" -a \
end="$END" -s mongodb_url="$MONGODB_URL" \
-s proxy_file_path="proxy-list.txt" -s use_cached_proxy="$USE_CACHED_PROXY"
