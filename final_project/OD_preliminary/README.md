# YOLO v8 標記結果
不是最終的YOLO模型跟資料集哦!

## 資料來源

- **TrainA, TestA**: 有痘痘的照片（挑過），來自 [Kaggle Dataset](https://www.kaggle.com/datasets/madurihansanie/acneo4dataset?resource=download)。

- **YOLO模型訓練資料**: 可在 Roboflow Universe [acne-new Image Dataset](https://universe.roboflow.com/buyumedatasets/acne-new/dataset/36) 上找到。(不重要)

## 標記資料夾
- **`TrainA_label`, `TestA_label`**: YOLO模型的標記結果 (.txt格式)。

---

## 標記資料欄位說明

- **x**: 物件中心 x 位於整張圖片寬度的比例。
- **y**: 物件中心 y 位於整張圖片高度的比例。
- **w**: 物件的寬度，佔整張圖片寬度的比例。
- **h**: 物件的長度，佔整張圖片長度的比例。
- **conf**: YOLO 偵測到痘痘的機率，我把大於0的全部output了，可以自己設定threshold~

詳細說明可參考 [這篇文章](https://blog.cavedu.com/2019/07/25/yolo-identification-model/)。

---

## Conf 挑選 Example

- **原始照片**
![原始照片](Example/levle0_1.jpg)

- **conf > 0.01**:
  ![conf > 0.01](Example/levle0_1_0.01.jpg)

- **conf > 0.1**:
  ![conf > 0.1](Example/levle0_1_0.1.jpg)

- **conf > 0.2**:
  ![conf > 0.2](Example/levle0_1_0.2.jpg)
