# DL_HW2
## Data preprocess
我根據imagenet-mini中所給的txt檔案，將所有圖片分入三個folders，分別為train, val, test，而其中又依照類型去做分類，所以三個資料夾中又都分別有50個不同的資料夾。  
對於所有的圖片都先將一邊resize到256，再center crop成為224*224。  
## Question 1
第一題我用resnet18當作測驗的model，我分別訓練了有加入dynamic convolution的model與沒有加入dynamic的model。  
兩個model分別訓練了30epoch，並且比較結果。  
![image](https://hackmd.io/_uploads/r1XZHibUC.png)
我們可以看到兩種方法的loss、accuracy的比較。訓練時，dynamic model的loss較original的model下降的慢，但是慢的非常微小，反而在validation時，loss的表現較original的更好。  
至於accuracy的表現，training時相差無幾，validation時又是dynamic以些許的優勢略勝一籌。  


| Resnet18 | original |  dynamic |
| -------- | -------- | -------- |
| Test acc |   0.6267 |  0.6467  |
| params(M)|   11.2   |  44.9    |
| MACs(M)  |   1820   |  178.3   |
| one epoch(s)  |   80  |  185   |

藉由上面的表格的Test accuracy，我們能夠看到dynamic convolution ResNet18確實以非常微小的優勢勝過original ResNet18，並且MACs也比original的要小，但是結果就是模型比較臃腫，為original的整整四倍，且在電腦上的訓練時間也是original的兩倍以上。  

結論就是雖然dynamic有較好的performance，但是也是有效率上的一些犧牲作為交換，這就是模型的trade-off。  


## Question 2
一開始先訓練ResNet34，設定為batch size = 128, learning rate = 0.001, crossentropy, adam，並且訓練10epoch。  
測量出的Test accuracy為0.628 ![image](https://hackmd.io/_uploads/Sk57Vp-80.png)



第一個設計的model為alexnet的revised，我將其中一層的convolution layer刪除，所以共有4層conv layer與2層FCs(不包含最後一層)。
獲得的結果為大約為0.02，出乎意料外的低。

第二個設計的model的backbone為bottelneck的residual block，整體的架構是先放入一層的conv layer，再加上三層的bottelneck resblock，最後放入一層FC(不包含output層)。
獲得的結果為0.482 ![image](https://hackmd.io/_uploads/SkRCQT-8R.png)  

最後一個設計參考了mobilenetV3中的basic block，類似上面的操作，先1個conv layer，接上三個basic block，最後在接上一個FC。
獲得的結果為0.29 ![image](https://hackmd.io/_uploads/H11VKpZL0.png)

將所有的結果畫成圖片
![image](https://hackmd.io/_uploads/r17UtTb8A.png)

## 討論
對於第一題Dynamic Convolution ResNet18 在性能上略優於原始 ResNet18，但在效率上需要付出一定的代價。這是一個典型的trade-off問題。雖然 Dynamic Convolution 能夠在某些場景下提高準確率並降低計算量，但它會顯著增加模型的參數量和訓練時間。因此，在選擇是否使用 Dynamic Convolution 時，需要根據具體應用場景來平衡性能和效率。  

對於第二題，在這些實驗中，ResNet34作為base model，性能最佳，這可能是因為它的結構設計更為成熟和經過驗證。相比之下，修改後的 AlexNet 模型和使用 MobileNetV3 basic blcok的模型性能不佳，這表明這些結構在處理這個特定數據集和任務時可能並不合適。

使用 bottleneck residual block的模型性能略好於其他設計，但仍然低於 ResNet34。這可能是因為 bottleneck resblock需要更深的網絡來充分發揮其效果，而這裡僅使用了三層。

總體來說，這些實驗強調了模型設計的重要性。雖然新的架構可能在某些情況下有效，但在實踐中，成熟的、經過驗證的模型架構（如 ResNet34）往往能提供更穩定和優越的性能。

## Reference
https://github.com/TArdelean/DynamicConvolution/tree/master/models  
https://github.com/WZMIAOMIAO  
https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch


