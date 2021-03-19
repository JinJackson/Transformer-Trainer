# Simple Instruction for 'Trainer' in Transformers

### dataset: IMDB

You can download it from [here][http://ai.stanford.edu/~amaas/data/sentiment/]

+ Train data: 25000

  + pos:12500
  + neg:12500

  --> use 80% for Training and 20% for Dev

+ Test data: 25000

  + pos:12500
  + neg:12500

### File Structure

+ /model 
  + Contains Model file which uses Bert [cls] and a Linear layer
+ /result
  + Save checkpoint and result record
+ dataset.py
  + Data Preprocess and make it as Dataset of torch.utils.data
+ parser1.py
  + Input hyper parameters
+ Train.py
  + Training File