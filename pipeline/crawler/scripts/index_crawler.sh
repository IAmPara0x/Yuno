#!/bin/bash

for ARGUMENT in "$@"
  do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
      CREATE_PROXY) CREATE_PROXY=${VALUE};;
      PARSE_RECS) PARSE_RECS=${VALUE};;
      MONGODB_URL) MONGODB_URL=${VALUE} ;;
      USE_CACHED_PROXY) USE_CACHED_PROXY=${VALUE} ;;
      PROXY_FILE_PATH) PROXY_FILE_PATH=${VALUE} ;;
      *)
    esac
  done

if [[ ! -n "$PROXY_FILE_PATH" ]]
then
  PROXY_FILE_PATH="data/proxy-list.txt"
fi

if [ "$CREATE_PROXY" == "True" ];
then
timeout --signal=SIGINT 30 proxybroker find --types HTTPS > data/proxy-list.txt
./preprocess.py
fi


scrapy crawl index-crawler -a parse_recs="$PARSE_RECS" \
-s mongodb_url="$MONGODB_URL" \
-s proxy_file_path="$PROXY_FILE_PATH" \
-s use_cached_proxy="$USE_CACHED_PROXY"
