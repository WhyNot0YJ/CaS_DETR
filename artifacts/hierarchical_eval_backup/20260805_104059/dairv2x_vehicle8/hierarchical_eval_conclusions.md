# dairv2x_vehicle8 层级合并评测结论

结论仅使用 test 原始指标；未构造综合分数。所有结果均为原生后处理后的 label-collapse-only COCOeval。

## 总体最佳检测器

- AP50：root_dfine_dairv2x，0.922165
- AP50:95：root_dfine_dairv2x，0.672643
- APs50：cas_deim_moe4_only_hgnetv2_s_dairv2x，0.753627
- APs50:95：root_dfine_dairv2x，0.441340

## 基线最佳

- AP50：root_dfine_dairv2x，0.922165
- AP50:95：root_dfine_dairv2x，0.672643
- APs50：root_dfine_dairv2x，0.752294
- APs50:95：root_dfine_dairv2x，0.441340

## CaS-DETR 最佳

- AP50：cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x，0.921616
- AP50:95：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.672066
- APs50：cas_deim_moe4_only_hgnetv2_s_dairv2x，0.753627
- APs50:95：cas_deim_moe_cass_keep05_hgnetv2_s_dairv2x，0.440282

## 组件消融最佳版本

- AP50：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.921389
- AP50:95：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.672066
- APs50：cas_deim_moe4_only_hgnetv2_s_dairv2x，0.753627
- APs50:95：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.438178

## dynamic base ratio 最佳版本

- AP50：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.921389
- AP50:95：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.672066
- APs50：cas_deim_moe_cass_caip_base05_a10_hgnetv2_s_dairv2x，0.752349
- APs50:95：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.438178

## fixed keep ratio 最佳版本

- AP50：cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x，0.921616
- AP50:95：cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x，0.671955
- APs50：cas_deim_moe_cass_keep05_hgnetv2_s_dairv2x，0.752256
- APs50:95：cas_deim_moe_cass_keep05_hgnetv2_s_dairv2x，0.440282

## MoE 容量扫描最佳版本

- AP50：cas_deim_moe4_cap2x_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.919241
- AP50:95：cas_deim_moe4_cap2x_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.668706
- APs50：cas_deim_moe4_cap2x_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.749087
- APs50:95：cas_deim_moe4_cap2x_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.436888

## 专家数扫描最佳版本

- AP50：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.921389
- AP50:95：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.672066
- APs50：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.751984
- APs50:95：cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x，0.438178

## 独立根目录 checkpoint 最佳版本

- AP50：root_dfine_dairv2x，0.922165
- AP50:95：root_dfine_dairv2x，0.672643
- APs50：root_dfine_dairv2x，0.752294
- APs50:95：root_dfine_dairv2x，0.441340
