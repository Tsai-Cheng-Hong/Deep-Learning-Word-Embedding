# Deep-Learning-Word-Embedding
NLP-各式詞向量的訓練方法與種類

Gensim的版本為3.8.3 

pip install gensim==3.8.3

我們使用的Dataset是維基百科語料庫

------------------------------------------------------------------------
# 語言模型的訓練前處理
訓練中文的語言模型前，需要先對Dataset進行前處理

所需要進行的前處理有:1.簡體字翻譯中文字 2.斷詞(Word Segmentation)

1.繁體字翻譯中文字:https://github.com/Tsai-Cheng-Hong/Python-OpenCC-zh-cn_translate_to_zh-tw

2.斷詞(Word Segmentation):https://github.com/Tsai-Cheng-Hong/CKIP-Transformers-Word-Segmentation

------------------------------------------------------------------------
# 詞向量的評估
語言模型的評估方式可以使用詞相似度(Word Similarity)與詞類比(Word Analogy)

通過計算關聯性的分數，來了解模型訓練的好壞

評估方式 : https://github.com/Tsai-Cheng-Hong/Deep-Learning-Word-Embedding-evaluation

------------------------------------------------------------------------
# 詞向量的應用
語言模型訓練出來的詞向量可以做許多不同的後續任務

可以拿來應用在文件分類、語意分析...等任務上

為了讓這些下游任務有良好的成效，因此好的語言模型十分重要

有好的語言模型，相當於人類擁有好的字典

文件分類任務:https://github.com/Tsai-Cheng-Hong/Deep-Learning-Document-Classification

語意分析任務:https://github.com/Tsai-Cheng-Hong/Deep-Learning-Semantic-Analysis

------------------------------------------------------------------------
# 已經訓練好的繁體中文詞向量
Dataset為維基百科語料庫

Gensim-CBOW:https://drive.google.com/drive/folders/1UxMw4Gt6mSHXdpzIAX01MTvAhcIcfba2?usp=sharing

Gensim-SkipGram:https://drive.google.com/drive/folders/1C8zw1my3ZQG6LiPprvKroTPsbw3sTNJN?usp=sharing

Gensim-FastText:https://drive.google.com/drive/folders/150Jw_voKXT-l048Qjpl8y0_bp7GO2C8n?usp=sharing

CWE:https://drive.google.com/file/d/16-8GsaHZvOG3RAXOhYDGzMlboWqxHrO2/view?usp=sharing

JWE:https://drive.google.com/file/d/1F9hvpjrb3AhLdfnfNzBM9lFlwlTXNDf9/view?usp=sharing

GWE:https://drive.google.com/file/d/1cu4Y2JfAc7JyBvj_W6rjD8GnaiGrBR2c/view?usp=sharing

SCWE:https://drive.google.com/file/d/1YLQe46eQNuVouOvNeVd7P9jZCWjDPcXA/view?usp=sharing

AWE-self:https://drive.google.com/file/d/1CvEo6gHruynUXk46IFjMnGCQsaqTzTQ0/view?usp=sharing

AWE-global:https://drive.google.com/file/d/1ZLl79Bs2sXehZJskEH5CPO8jLA_BdLxq/view?usp=sharing

P&AWE-global:https://drive.google.com/file/d/1gq9sg9-41pnoyPdcU5LdIVPdzs87M3os/view?usp=sharing

