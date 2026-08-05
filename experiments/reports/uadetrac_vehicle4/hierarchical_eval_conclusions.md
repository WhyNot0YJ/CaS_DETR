# uadetrac_vehicle4 层级合并评测结论

结论仅使用 test 原始指标；未构造综合分数。所有结果均为原生后处理后的 label-collapse-only COCOeval。

## 总体最佳检测器

- AP50：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.895917
- AP50:95：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.671590
- APs50：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.704722
- APs50:95：root_dfine_uadetrac，0.429316

## 基线最佳

- AP50：root_deim_uadetrac，0.879960
- AP50:95：root_deim_uadetrac，0.661505
- APs50：root_deim_uadetrac，0.687169
- APs50:95：root_dfine_uadetrac，0.429316

## CaS-DETR 最佳

- AP50：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.895917
- AP50:95：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.671590
- APs50：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.704722
- APs50:95：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.428917

## 组件消融最佳版本

- AP50：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.891099
- AP50:95：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.671408
- APs50：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.704722
- APs50:95：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.428917

## dynamic base ratio 最佳版本

- AP50：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.891099
- AP50:95：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.671408
- APs50：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.704722
- APs50:95：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.428917

## fixed keep ratio 最佳版本

- AP50：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.895917
- AP50:95：cas_deim_moe3_cass_keep05_hgnetv2_s_uadetrac，0.671590
- APs50：cas_deim_moe3_cass_keep07_hgnetv2_s_uadetrac，0.683580
- APs50:95：cas_deim_moe_cass_keep07_hgnetv2_s_uadetrac，0.423075

## MoE 容量扫描最佳版本

该数据集无对应实验。

## 专家数扫描最佳版本

- AP50：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.891099
- AP50:95：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.671408
- APs50：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.704722
- APs50:95：cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac，0.428917

## 独立根目录 checkpoint 最佳版本

- AP50：root_deim_uadetrac，0.879960
- AP50:95：root_deim_uadetrac，0.661505
- APs50：root_deim_uadetrac，0.687169
- APs50:95：root_dfine_uadetrac，0.429316
