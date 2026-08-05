# uadetrac_vehicle4 细粒度评测结论

结论仅使用 test 原始指标；未构造综合分数。所有结果均为原生后处理后的 label-collapse-only COCOeval。

## 按类别 AP50 最佳

- car：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.878779
- van：root_dfine_uadetrac，0.780315
- bus：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.966442
- others：cas_deim_moe3_cass_keep07_hgnetv2_s_uadetrac，0.867878

## 按类别 AP50:95 最佳

- car：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.648612
- van：cas_deim_all_off_hgnetv2_s_uadetrac，0.630417
- bus：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.808474
- others：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.728590

## 按尺度 AP50 最佳

- small：cas_deim_moe3_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.415754
- medium：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.761671
- large：cas_deim_all_off_hgnetv2_s_uadetrac，0.960442

## 按尺度 AP50:95 最佳

- small：cas_deim_moe3_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.272413
- medium：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.581079
- large：cas_deim_all_off_hgnetv2_s_uadetrac，0.813791

## 按天气 AP50 最佳

- cloudy：root_deim_uadetrac，0.861027
- night：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.792414
- rainy：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.904064
- sunny：cas_deim_all_off_hgnetv2_s_uadetrac，0.887165

## 按天气 AP50:95 最佳

- cloudy：root_deim_uadetrac，0.725367
- night：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.639213
- rainy：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.730787
- sunny：cas_deim_all_off_hgnetv2_s_uadetrac，0.683382

## CaS-DETR vs 基线（test split）

- AP50：cas_deim_moe3_cass_keep05 0.868131 vs root_deim 0.850747（+1.74%）
- AP50:95：cas_deim_moe4_cass_caip_base05_a10 0.697926 vs root_deim 0.683931（+1.40%）
- APs50：cas_deim_moe4_cass_caip_base05_a10 0.704722 vs root_deim 0.687169（+1.76%）
- APs50:95：root_dfine 0.429316 vs cas_deim_moe4_cass_caip_base05_a10 0.428917（-0.04%）

## YOLO/Faster R-CNN 最佳参考

- AP50：fasterrcnn_resnet50_fpn，0.803183
- AP50:95：yolox_m，0.627586
- APs50：yolov5s，0.320737
- APs50:95：yolov5s，0.197258
