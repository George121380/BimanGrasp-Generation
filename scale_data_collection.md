Multi-GPU :
```bash
python tools/scale_runner.py \
  --exp_name cthulhu1 \
  --gpus 0,1,2,3,4,5 \
  --object_code_list 101507 101517 101531 101541 101546 101557\
  --target_count_per_object 30000 \
  --round_batch_size 2048 \
  --num_iterations 10000 \
  --seed_base 0 \
  --resume \
  --final_dir_mode seed_only
```


done list:
 
d4 100017 100021 100023 100025 100028 100032
d5 102736 102761 102763 100015 (done)
d1 100031 101305 101315 102714 102715 102720 102724 102726 102732 (done)
cthulhu6 101336 101417 101442 101458 101490 101501
cthulhu1 101507 101517 101531 101541 101546 101557 (done)




```bash
python tools/batch_scale_from_json.py \
  --json_path data/objects_by_category.json \
  --meshroot data/meshdata \
  --exp_name toys \
  --gpus 1,2,4,5,6,7,8,9 \
  --target_count_per_object 1024 \
  --round_batch_size 1024 \
  --num_iterations 10000 \
  --seed_base 0 \
  --resume \
  --final_dir_mode both \
  --categories toys
```

d1 bottle_bowl bottles bowls
cthu1 clocks_containers clocks containers 
cthu3 displays_kettle displays kettle
cthu4 others_plates others plates
cthu6 pots_shoes pots shoes
d6 dispensor dispensor
d1 toys toys


```bash
python tools/batch_scale_from_json.py \
  --need_add_path data/object_need_add.json \
  --meshroot data/meshdata \
  --select_objects 100021 100623 100032 100028 100023 100025\
  --exp_name need_add_c6 \
  --gpus 0,1,2,3,4,5 \
  --target_count_per_object 12288 \
  --round_batch_size 2048 \
  --num_iterations 10000 \
  --seed_base 0 \
  --resume \
  --final_dir_mode both \
  --dry_run
```

d1 need_add_d1 Dixie_10_ounce_Bowls_35_ct Twinlab_100_Whey_Protein_Fuel_Cookies_and_Cream ddg-gd_jar_poisson_014 Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original 102715 102763 101305 102761 102714

cthu6 need_add_c6 100021 100623 100032 100028 100023 100025



# uni2bim版本
python tools/batch_scale_from_json.py \
  --json_path data/objects_by_category.json \
  --meshroot data/meshdata \
  --exp_name uni2bim_others_plates \
  --gpus 0,1,2,3,4,5 \
  --target_count_per_object 1024 \
  --round_batch_size 1024 \
  --num_iterations 10000 \
  --seed_base 0 \
  --resume \
  --final_dir_mode both \
  --categories others plates  \
  --mode uni2bim
```

d1 bottle_bowl bottles bowls
cthu1 clocks_containers clocks containers 
cthu3 displays_kettle displays kettle
cthu4 others_plates others plates
cthu6 pots_shoes pots shoes
d6 dispensor dispensor
d1 toys toys
