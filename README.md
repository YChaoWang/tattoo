# 刺青辨識
# 說明

## 資料集
刺青圖片：可以放在與README.md同一層，資料夾名稱為image_1（也可以用其他名稱，但這樣程式碼也要記得修改）


## VGG 比對
- 檔案
1. VGG沒對齊相同 -> vgg_compare/vgg_same_folder.py
2. VGG沒對齊不同 -> vgg_compare/vgg_different_folder.py
3. VGG對齊相同 -> vgg_compare/align/vgg_same_folder_aligned.py
4. VGG對齊不同 -> vgg_compare/align/vgg_different_folder_aligned.py
- 執行範例 VGG沒對齊相同
```bash
python3 vgg_compare/vgg_same_folder.py
```

## SVM
- 檔案
1. 訓練 -> svm/訓練/svm_similarity.py (資料量大時訓練時間較久) 或是 svm/訓練/svm_similarity_linearsvc.py
2. 預測 -> svm/預測/svm_only_similarity_formula.py
- 執行範例 SVM預測
```bash
python3 svm/svm_only_similarity_formula.py
```
## SuperGlue 特徵點配對
- 檔案
1. SuperGlue對齊相同 -> superglue_compare/superglue_same_folder_aligned.py
2. SuperGlue對齊不同 -> superglue_compare/superglue_different_folder_aligned.py
- 執行範例
```bash
python3 superglue_compare/superglue_same_folder_aligned.py 
```

## 對齊+VGG篩選資料夾+SuperGlue特徵點配對
- 檔案
tattoo_all_flows.py
- 執行範例
```bash
python3 tattoo_all_flows.py
```