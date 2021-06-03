# wide_deep

# wide-deep recommendation system for Movielens with tensorflow 2

## --reference paper
Cheng, Heng-Tze & Koc, Levent & Harmsen, Jeremiah & Shaked, Tal & Chandra, Tushar & Aradhye, Hrishi & Anderson, Glen & Corrado, G.s & Chai, Wei & Ispir, Mustafa & Anil, Rohan & Haque, Zakaria & Hong, Lichan & Jain, Vihan & Liu, Xiaobing & Shah, Hemal. (2016). Wide & Deep Learning for Recommender Systems. 7-10. 10.1145/2988450.2988454. 


## --description
+ dataset : Movielens
+ predict sentiments 0 ~ 3.5 == > class "0".    4.0~5.0 ==> class "1"
+ make cross-feature : genre x year
+ feature added : tag sentiment score, # of rated user, movie


## example : FM
```
python fm.py --path "./datasets/" --dataset "movielens" --layers [1024,512,256] --epochs 10 --test_size 0.1 --batch_size 32 --deep_regs [0,0,0] --lr 0.001 --learner 'adam' --out 1 --patience 10

```
