# Bang so lieu (tu sinh, khong sua tay)

## 1. Val va test (conf 0,25)

| eval_set | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| test | 0,854 | 0,934 | 0,972 | 0,516 |
| val | 0,975 | 0,958 | 0,978 | 0,551 |

## 2. Box thua theo loai

| tap | FP | trung | lech | nen |
|---|---|---|---|---|
| val | 20 | 10 | 8 | 2 |
| test | 307 | 2 | 6 | 299 |

## 3. Quet conf tren val

| conf | P | R | F1 | box_sai | bo_sot |
|---|---|---|---|---|---|
| 0.25 | 0,962 | 0,977 | 0,970 | 23 | 11 |
| 0.4 | 0,976 | 0,954 | 0,965 | 14 | 25 |
| 0.5 | 0,981 | 0,942 | 0,961 | 11 | 33 |
| 0.6 | 0,987 | 0,905 | 0,945 | 7 | 56 |

## 4. Nguong da chon (tren val)

Nguong chung: 0.25

| lop | conf |
|---|---|
| Missing Hole | 0.25 |
| Mouse Bite | 0.25 |
| Open Circuit | 0.25 |
| Short | 0.5 |
| Spur | 0.25 |
| Spurious Copper | 0.25 |

## 5. Box thua tren test theo vi tri (tam, lam tron 30 px; >= 5 anh)

| bo | pred_class | cx | cy | anh | conf_tb |
|---|---|---|---|---|---|
| 08 | Missing Hole | 180 | 1080 | 60 | 0.59 |
| 08 | Missing Hole | 1740 | 510 | 59 | 0.34 |
| 12 | Short | 1170 | 1140 | 60 | 0.63 |
| 12 | Short | 1560 | 1110 | 60 | 0.39 |
| 12 | Spurious Copper | 1710 | 1560 | 51 | 0.32 |

290/307 FP nam o 5 vi tri lap lai; 16 vi tri con lai la le te.
