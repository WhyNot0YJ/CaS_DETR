# dairv2x_vehicle8 细粒度评测结论

结论仅使用 test 原始指标；未构造综合分数。所有结果均为原生后处理后的 label-collapse-only COCOeval。⚠️ cas_deim_moe_cass_caip_base05_a0_fixed_hgnetv2_s_dairv2x 仅训练 21 epoch，已排除。

## 按类别 AP50 最佳

- car：cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x，0.972819
- truck：cas_deim_moe_cass_caip_base05_a10_hgnetv2_s_dairv2x，0.835752
- van：cas_deim_moe_cass_keep05_hgnetv2_s_dairv2x，0.885188
- bus：cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x，0.979332
- pedestrian：cas_deim_moe4_cass_caip_base03_a10_keep07_fixed_hgnetv2_s_dairv2x，0.903096
- cyclist：cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x，0.915946
- motorcyclist：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.850445
- trafficcone：cas_deim_moe_cass_caip_base05_a10_hgnetv2_s_dairv2x，0.966429

## 按类别 AP50:95 最佳

- car：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.842635
- truck：cas_deim_moe_cass_caip_base05_a10_hgnetv2_s_dairv2x，0.570200
- van：cas_deim_moe_cass_keep05_hgnetv2_s_dairv2x，0.782922
- bus：cas_deim_moe_cass_caip_base05_a10_hgnetv2_s_dairv2x，0.869359
- pedestrian：cas_deim_moe4_cass_caip_base03_a10_keep07_fixed_hgnetv2_s_dairv2x，0.575944
- cyclist：cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x，0.686180
- motorcyclist：root_dfine_dairv2x，0.622503
- trafficcone：root_dfine_dairv2x，0.639873

## 按尺度 AP50 最佳

- small：cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x，0.666168
- medium：cas_deim_moe_cass_caip_base05_a10_hgnetv2_s_dairv2x，0.910739
- large：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_dairv2x，0.971342

## 按尺度 AP50:95 最佳

- small：cas_deim_moe3_dim128_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.408769
- medium：cas_deim_moe_cass_caip_base05_a10_hgnetv2_s_dairv2x，0.687585
- large：root_dfine_dairv2x，0.871614

## CaS-DETR vs 基线（test split）

- AP50：cas_deim_moe_cass_caip_base05_a10 0.912518 vs root_dfine 0.909904（+0.26%）
- AP50:95：cas_deim_moe4_cass_caip_base03_a10 0.696354 vs root_dfine 0.695028（+0.13%）
- APs50：cas_deim_moe4_only 0.621439 vs root_dfine 0.542698（+7.87%）
- APs50:95：cas_deim_moe3_dim128_cass_caip_base03_a10 0.408769 vs root_deim 0.363074（+4.57%）

## YOLO/Faster R-CNN 最佳参考

- AP50：yolox_m，0.890729
- AP50:95：yolox_m，0.660822
- APs50：yolox_m，0.604384
- APs50:95：yolox_m，0.350283

---

## 8 已排除实验

| run_id | 排除原因 |
|--------|----------|
| cas_deim_moe_cass_caip_base05_a0_fixed_hgnetv2_s_dairv2x | 训练仅完成 21/100 epoch，数据不充分 |
