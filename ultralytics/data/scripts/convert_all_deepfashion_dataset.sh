bash ultralytics/data/scripts/get_consumer_to_shop_Clothes_Retrieval_Benchmark.sh  && python ultralytics/convert_deepfashion.py \
  --dataset-root "/root/SmartScan/datasets/consumer_to_shop_Clothes_Retrieval_Benchmark" \
  --images-root "/root/SmartScan/datasets/img" \
  --save-dir "/root/SmartScan/datasets/deepfashion_yolo"