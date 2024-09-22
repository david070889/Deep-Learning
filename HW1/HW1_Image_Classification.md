# DL_HW1_Image_Classification
## Introduction
這次作業使用了1. color histogram 2. ORB (Oriented FAST and Rotated BRIEF) 3. BRISK(Binary Robust Invariant Scalable Keypoints)。  
模型方面使用了SVM, RandomForest 和 KNN。  

**color histogram**為其中最簡單的特徵提取方法，先將圖片轉為灰階，並將灰階分成0-255、共256階的灰度，並統計每張片的各個灰階的數量，轉換成為特徵。  
  
**ORB**是一種融合了FAST和BRIEF技術的特徵提取算法。它首先利用FAST技術挑選出候選特徵點，這些點被選出來是基於其周圍有足夠多的像素點與候選點在灰度值上存在顯著差異。然後，ORB使用BRIEF算法對這些關鍵點周圍的像素進行描述和編碼。但由於BRIEF本身不具旋轉不變性，ORB對其進行了改進，通過根據關鍵點的方向旋轉BRIEF的描述算子，使其具有對圖像旋轉的不變性。雖然利用FAST可以找到大量關鍵點，但這些並非全部都是最優的。ORB最後會使用Harris score，篩選出最強的N個特徵點，以達到更高的準確度和效率。

**BRISK**是一種特徵點檢測和描述的方法，它採用了一種快速的尺度空間近似技術。該技術通過在多個尺度層次上建立圖像，提高了對尺度變化的不變性，使得特徵點可以在不同尺度下被檢測。BRISK在特徵點檢測上基於改良的FAST算法，考慮到了尺度空間的因素，它在各個尺度層次上尋找corner，並選出那些在多尺度中均表現出corner特性的點作為關鍵點。為了給這些關鍵點賦予方向，BRISK計算了每個關鍵點周圍區域內的局部梯度方向直方圖，並以最大方向作為關鍵點的方向，實現旋轉不變性。其描述符是通過比較關鍵點周邊預定義模式上點對的相對亮度來生成的，這種方法支持使用Hamming距離進行高效的配對。

**SVM**是一種在分類和迴歸分析中使用的監督學習模型及其相關演算法。這個模型通過分析一組訓練數據，其中每個數據點都已被標記為屬於兩類中的一類，來構建一個預測模型。該模型作為一個非機率的二元線性分類器，目標是找到一個決策邊界，這樣可以將新的數據點分配給這兩個類別中的一個。在空間中，每個數據點被視為一個點，使得不同類別的點之間有最大的間隔。新數據點根據它們在這個空間中的位置—特別是它們相對於決策邊界的位置—被分類到某一類別中。

**KNN**是一種用於分類和迴歸的無母數統計方法，採用向量空間模型來分類，概念為相同類別的案例，彼此的相似度高，而可以藉由計算與已知類別案例之相似度，來評估未知類別案例可能的分類。  

**RandomForest**隨機森林是一種集成學習方法，通過建立多個決策樹並將它們的預測結果結合起來以提高預測準確性和控制過擬合，這種方法在決策樹的隨機選擇的特徵子集上進行訓練，並通過投票（對於分類問題）或平均（對於回歸問題）的方式來提升模型的性能。  


## Methodology

For **color histogram** :  
```python=
im1 = cv2.calcHist([im1], [0], None, [256], [0, 256])
```
函數 cv2.calcHist 用於計算圖像im1的灰階色彩直方圖，其中[im1]表示要處理的圖像列表，[0]指定從圖像的第一個通道（灰階圖像只有一個通道）計算直方圖， None 為mask參數，在這裡未使用表示對整張圖像進行處理， [256] 定義了直方圖箱子的數量，即有256個可能的像素值， [0, 256] 指定了像素值的範圍，這裡從0（黑色）到256（白色，但按照慣例不包括256），這段code的結果im1將是一個包含了圖像中每個像素值頻率統計的數組。  

For **ORB** :  
```python=
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(im1, None)
if descriptors is not None:
    imgs.append(np.mean(descriptors, axis=0))
else:
    imgs.append(np.zeros(32))
```

這段code首先將圖像im1從BGR色彩空間轉換為灰階，這是因為ORB演算法在提取特徵時只需要圖像的亮度信息。隨後，使用cv2.ORB_create()創建ORB detector。orb.detectAndCompute(im1, None) 調用用於在灰階圖像上檢測keypoints並為每個keypoints計算描述符，其中第二個參數None表示沒有使用mask，即考慮圖像的所有區域。這時計算所有descriptor的平均值並將結果添加到列表imgs中；如果沒有檢測到keypoints，即descriptor為 None，則在列表中添加一個長度為32的零向量。這樣做的目的是保持特徵向量的維度一致，便於後續處理。這裡選擇32作為零向量的大小是基於ORB描述符的典型維度，雖然ORB描述符的默認長度是256位。  

For **BRISK** :  
```python=
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
brisk = cv2.BRISK_create() 
keypoints, descriptors = brisk.detectAndCompute(im1, None)
if descriptors is not None:
    imgs.append(np.mean(descriptors, axis=0))
else:
    imgs.append(np.zeros(64))  
```

在這段code中，首先使用cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) 將彩色圖像im1轉換成灰階圖像，以便進行BRISK特徵提取。接著，通過 cv2.BRISK_create()創建了一個特徵檢測器brisk。使用 brisk.detectAndCompute(im1, None)方法在灰階圖像上同時檢測keypoints 並計算它們的descriptors。這裡，None 參數表示沒有使用mask，所以對整個圖像進行操作。如果檢測到的 descriptors 不為空，則對 descriptors 數組沿著每個descriptors的特徵維度（axis=0）計算平均值，並將結果陣列添加到列表 imgs 中。這個平均過程旨在從每個keypoint 的descriptor中提取一個固定長度的特徵向量。如果沒有檢測到descriptor（即為None），則向 imgs 列表中添加一個長度為64的零向量，作為該圖像特徵的佔位符。這樣確保了無論是否檢測到描述符，imgs 列表中對於每張圖像都有一個固定長度的特徵表示，方便後續進行機器學習或其他處理。  

## Experiment & result
|特徵提取|histogram|ORB|BRISK|
|----|----|----|----|
|SVM|4%|4%|3%|
|KNN|1%|4%|2%|
|Random Forest|1%|3%|2.5%|

根據實驗統整的表格，我們可以看到對於三種不同的特徵提取方法：灰階直方圖、ORB和BRISK，它們在搭配三種不同分類器時的accuracy有所不同。對於SVM 和KNN分類器，ORB特徵提取的accuracy均為4%，而使用 Random Forest分類時，accuracy略低於3%。這表明ORB特徵提取與這些分類器相結合時提供了一定程度的一致性和穩定性。  

相比之下，灰階直方圖在搭配SVM時的accuracy為4%，但在KNN和隨機森林分類器中accuracy降至1%，這可能意味著直方圖特徵與基於距離的分類器更為適合。BRISK特徵提取在所有三種分類器中均呈現出最低的accuracy，這顯示了其在這些特定情境下的優勢。SVM與BRISK的組合accuracy為3%，而KNN和隨機森林分類器分別有2%和2.5%的accuracy，表明BRISK特徵對於減少模型預測錯誤具有一定的效果。  

## Discussion & Conclusion

在我們的討論中，我們探討了多種特徵提取技術及其在不同機器學習模型中的應用。從灰階色彩直方圖到ORB和BRISK，每種方法都有其獨特的應用場景和效益。灰階色彩直方圖通過統計圖像中的亮度分佈提供了一種有效的方式來分析和處理圖像。而ORB算法則將關鍵點檢測和二進制描述符的計算結合起來，提供了對圖像旋轉的不變性，適合於需要快速處理的場合。BRISK作為一種更加先進的特徵提取方法，提供了對尺度和旋轉的不變性，並在計算效率上進行了優化。

通過分析不同特徵提取方法在配合SVM、KNN和隨機森林等分類器的效果，我們發現ORB特徵在與不同的分類器組合時表現出了穩定的accuracy，顯示其在各種情況下都能保持一致的性能。相比之下，灰階色彩直方圖在KNN和隨機森林分類器中的表現尤為突出，這可能是由於這些模型能夠更好地利用基於距離的特徵。BRISK特徵在所有分類器中都實現了較低的accuracy，特別是在與SVM結合時，顯示了其對於減少預測正確的潛力。

總的來說，這些發現強調了選擇合適特徵提取技術和機器學習模型的重要性，因為不同的組合會對模型的性能產生顯著影響。在實際應用中，正確的配對可以提高準確率，降低錯誤率，並在特定的任務中實現最佳的結果。因此，在開發機器學習系統時，應該通過交叉驗證來細心調整特徵提取方法和分類器的選擇，以達到最優的性能。