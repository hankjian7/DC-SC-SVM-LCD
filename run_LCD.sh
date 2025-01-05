###kitti00###
./build/LCDEngine --parameters /root/LCD/LCDEngine/params.yml\
    --img_list /root/LCD/LCD_data/kitti00/image_list_2.txt\
    --des-path /root/LCD/LCD_data/des9_bin/kitti00\
    -o /root/LCD/LCD_data/experimental_result/kitti00/original_with_dual_with_scch_no_svm_topk2.txt\
    --topk 2\

###CC###
./build/LCDEngine --parameters /root/LCD/LCDEngine/params.yml\
    --img_list /root/LCD/LCD_data/CC/left_image_list.txt\
    --des-path /root/LCD/LCD_data/des9_bin/CC\
    -o /root/LCD/LCD_data/experimental_result/CC/tmp_final.txt\
    --topk 2\