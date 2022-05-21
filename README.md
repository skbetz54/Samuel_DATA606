# The Fear-Industrial Complex<sup>1</sup>
*An analysis on the amount of fear that plays into current US news media by Samuel Kolodrubetz*

## Links
---

- [Link to full proposal](https://github.com/skbetz54/Samuel_DATA606/blob/main/Approved_Proposal.md)
- [Link to proposal video presentation](https://www.youtube.com/watch?v=F_P-RPXwhEE)
- [Link to proposal presentation](https://github.com/skbetz54/Samuel_DATA606/blob/main/Samuel%20Kolodrubetz%20-%20Fear%20Industrial%20Complex.pptx)
- [Link to EDA](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/EDA_Hugging_Face.ipynb)
- [Link to Machine Learning](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Machine_Learning_Implementation.ipynb)
- [Link to Web Scrape - CNN](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Web_Scrape_CNN.ipynb)
- [Link to Web Scrape - Fox News](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Web_Scrape_Fox_News.ipynb)
- [Link to Article Testing](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Articles_Testing.ipynb)
- [Link to final presentation PPT](https://github.com/skbetz54/Samuel_DATA606/blob/main/Data606%20Samuel%20Kolodrubetz%20-%20Final%20Presentation.pptx)
- Link to final presentation video [part 1](https://youtu.be/V4Cn680XiuM) & [part 2](https://youtu.be/6K9PVBj8Z5M)




## Topic Introduction
---

Anyone familiar to American journalism and/or news media is familiar with the quote "if it bleeds, it leads" <sup>2</sup>. For those who aren't, this essentially tells us that popular and leading stories often involve blood, death, controversial subjects, and other similar topics all revolving around one thing: **fear**. Fear is nowhere near a new concept, especially in journalism. The above quote was coined in the late 19<sup>th</sup> century, and that feeling still endures today. If you turn on any local TV station or read any online article the top story is likely going to be something that involves violence, a scandal, or a similar salacious event. 

Motivations for each individual news outlet differs as well. Places like the nonprofit news agency AP News qualify themselves as an independent news source who pride themselves on unbiased, factual reporting. At the same time, other news sources that also claim to be committed to factual reporting have other reasons for their stories such as profit or political motivations. These added incentives for these sources to "spice up" their story to make the readers and watchers coming back for more, as more clicks or views normally means more influence or money. As noted by the above quote, one such tactic is the use of fear to scare a reader, but how much is it still being used?

## Questions to Answer
---

This project looks at two questions regarding the use of fear within (online) news topics:


1. Is it possible to quantify the amount of fear (among other emotions) in news articles among various news outlets? If so, how popular of a tactic is it?
2. Is there a difference in the amount of fear used among different large media outlets?


## Data Source and Description
---

The first step is using deep learning to train a Recurrent Neural Network (RNN) on labeled text data from [huggingface.co](https://huggingface.co/datasets/emotion). The dataset contains 416,806 sentences and tweets which are classified as as one of 6 emotions: sadness, joy, love, surprise, anger, and fear.

**Note:** This dataset is similar to [one found on Kaggle](https://www.kaggle.com/pashupatigupta/emotion-detection-from-text), but that dataset focuses on tweets with over 13 emotions, but this is a separate dataset with only 5. [Huggingface dataset github](https://github.com/dair-ai/emotion_dataset)

Once the model(s) is trained and tested to a sufficient accuracy, I am then able to perform the analysis of emotions being used in real-world articles. To do this, I'll first create a Pandas dataframe with the articles' date, headline, and text (obtained using the Python library Newspaper3k). In order to determine how fear and other emotions are used differently in different outlets, I will be considering two news outlets across the political spectrum: CNN and Fox News. Additonally, after some investigation the sensational languages being used occurr much more frequently within opinion articles. For this reason, articles scraped for this project are opinion pieces hosted on these sites.

The variable of measure in this project is the emotion of the article/headline/sentence. The main focus is going to be building a model that will be able to accurately classify an input string (tweet or article) as any of the 6 emotions based on the labeled huggingface dataset. This classification is done by using a softmax output for each input string, which assigns a list of probabilities to each emotion based on how likely it is to be the correct answer. The highest probability is then selected and used as the output's prediction. For example, a sentence reading "I am feeling sad" could have an output of (0.98, 0, 0, 0.01, 0.01), where the first value is the "saddness" class, which would return "Sadness" as the prediction. Another intended unit of measure to add a "fear index" among other emotions, which would be the associated Softmax probability of the fear class (even if it wasn't the highest likelihood).


## **Exploratory Data Aanalysis**
---

- [Link to Happyface EDA notebook](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/EDA_Hugging_Face.ipynb)

The first part of my project entails creating a machine learning model (specifically a Recurrent Neural Network) using the labeled Huggingface emotion data, but before I dive into the modeling, I need to understand the dataset a little better. This entails both understanding the entire dataset, and also the difference between classes within the dataset. For example, in the image below we see that sadness and joy have a much higher number of samples compared to other emotions. This tells us that in the model creation, we will have to deal with this class imbalance by downsampling the high-frequency emotions.

<p align = 'center'>
<img width="500" alt="frequency chart" src="https://user-images.githubusercontent.com/70443630/155910809-a578cbf8-e554-4b90-a0e8-38f26061dae0.PNG">
</p>

Another point of focus in this Exploratory Analysis is the word frequency for each emotion. When looking at the top 10 words used for fear and surprise, there is not much we can gather; there are only a couple words that are associated with the emotion (fear --> afraid, surprised --> shock). While this does not tell us much about unique words seen in each class, it can help us dictate what words NOT to include in the model building. For example, the word "really" and "like" appear often in every single class. Adding these types of words prevelant throughout the dataset for each class to the list of stopwords is something that I wanted to end up testing its impact.

<p align = 'center'>
<img width="500" alt="freq_fear" src="https://user-images.githubusercontent.com/70443630/155911189-b88e9b82-a42c-4eae-9f42-82157a3ee4e6.PNG"><img width="500" alt="freq_surprise" src="https://user-images.githubusercontent.com/70443630/155911193-9f31ffb9-d358-4b20-9cce-1c9c9632d849.PNG">
</p>
  
However, if we expand our search and include more of the top words being used for each emotion, we start to see a better picture of the unique vocabularies being used with the model's recognizing each emotion. To do this, I created wordclouds for each emotion and the frequency of words being used, with the top 100 words being shown.

First for fear:

<p align = 'center'>
<img width="600" alt="wordcloud_fear" src="https://user-images.githubusercontent.com/70443630/155911349-ca550de0-56d3-4340-b5a9-42a512b692ad.PNG">
<p>
  
And for surprised:

<p align = 'center'>
<img width="600" alt="wordcloud_surprised" src="https://user-images.githubusercontent.com/70443630/155911377-2932d035-b9d2-44ee-a088-b1cb031e2bab.PNG">
</p>
  
Both of these wordclouds show great examples of unique words that are present within these emotions. In the wordcloud for fear, words such as "nervious", "afraid", and "anxious" are likely to not be very prevalent within other emotions (at least for the more positive emotions). Similarly, surprise shows words that can only be asssociated with surprise such as "shock", "overwhelm", or "stun".

Lastly I took a look at the polarity of each individual emotion's dataset using Textblob's built in sentiment analysis library. This function assigns a polarity value between -1 and 1, where -1 is a more negative sentiment and 1 is the most positive sentiment. As we see from the below group of histograms, the negative emotions (top row) are all centered around 0 (neutral) with an obvious skew to the left towards a negative sentiment. On the other hand, the positive emotions (bottom row) are again centered around 0 but is skewed towards a more positive sentiment.

<p align = 'center'>
<img width="700" alt="hist_polarity" src="https://user-images.githubusercontent.com/70443630/169345576-e40743e1-048f-47db-ba18-e5a465c1d9f7.PNG">
</p>
  
## **Machine Learning Implementation**
---

- [Link to Machine Learning notebook](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Machine_Learning.ipynb)

### Data Preprocessing
---

There are a few steps required to allow Pytorch to work with the input data. Most of these steps are used for both the training and testing of our models on the huggingface tweet dataset and the testing of real-world articles:

1. *Tokenization* - Tokenization is the process of breaking the raw text data into smaller tokens, which is normally just the individual words found in the data.
2. *Stop Word Removal* - As mentioned above, stop words are words like "the" or "and" which do not contain much inherent value, and are removed from the tokenized list of words within the tweet or article. Different lists of stop words were added to the standard list because of problems with the dataset's nature (i.e. the data already removed apostrophes from the contractions, so contractions without the apostrophes needed to be added to the list of stop words).
3. *Lemmatization* - Lemmatization is the process of removing the endings of words and only leaving its base form, or the lemma. This process is similar to stemming but involves one more step: each full word is iterated through and assigned a part of speech. Then, using the English language lexicon database [WordNet](https://wordnet.princeton.edu/) to remove the word endings according to their part of speech.
4. *Dictionaries* - My dataset's dictionary contains the emotions contained within the dataset and all of the corresponding tweets associated with that emotion. This is an important step for creating my custom Pytorch dataset because it allows me to iterate through the dataset and assign the data and corresponding label to the Pytorch dataset object. 
5. *Vocabularies* - The vocabulary contains every word contained within the huggingface dataset and an associated index. This is again used in creating the Pytorch dataset by transforming the input list of words into a number, since Pytorch cannot work with raw text data.
6. *Pytorch Dataset Creation* - Pytorch gives its users the ability to easily store a dataset's samples and corresponding labels, as well as easy access to individual samples through the use of the data primitive Dataset. With my custom dataset built on top of this Pytorch Dataset, the dictionary is looped through to store the sample and label. Once these are stored, the vocabulary is used to transform the text into an integer. Finally, this list is transformed into a tensor, the data type needed to run through Pytorch models, which is a tuple containing the list of integers as well as the class value of the sample.

### Model Creation
---

Now that the data is cleaned and ready to be used within a deep learning model, I can now build a model to try and predict the emotions used within a real-world news article.

In order to create the best RNN to classify the news articles, I wanted to test various measures that could impact each model. The first is the input size of the dataset. To measure the impact I trained models on 3 different curated datasets:
1. The entire dataset of 416,806 tweets
2. A randomly sampled dataset with the two largest classes (love and joy) downsampled to 50,000 tweets each.
3. A randomly sampled with each class downsampled to 15,000 tweets each. 


Standard Dataset

<p align = 'center'>
<img width="400" alt="frequency chart - std" src="https://user-images.githubusercontent.com/70443630/168902050-67baa6a8-5447-4386-a58a-219f3e18d619.PNG">
</p>
  
50K Downsampled Dataset

<p align = 'center'>
<img width="400" alt="frequency chart - 50k" src="https://user-images.githubusercontent.com/70443630/168902097-5983101a-0346-46d7-9ef8-32c8f7fedf72.PNG">
</p>
  
15K Sample Datset

<p align = 'center'>
<img width="400" alt="frequency chart - 15k" src="https://user-images.githubusercontent.com/70443630/168902139-3d4ee574-e83c-4626-9f55-f94586f7dce0.PNG">
</p>


As we can see below, while the smaller datasets increased their accuracy slightly slower each epoch, the time taken for the full dataset was much longer. For that reason, the final "best" model was trained and tested on the dataset with 15,000 samples for each class. 
<p align = 'center'>
<img width="500" alt="b1_full_time" src="https://user-images.githubusercontent.com/70443630/168902624-b32fd28b-a2e2-4a8e-b02d-24d4f1a9c0fb.PNG">
</p>
The second hyperparameter of focus was batch size. Again, similar to the size of the dataset, the main difference in using different batch sizes is in training time. After testing various batch sizes (1, 16, & 32), the "best" model was trained and tested with a batch size of 32. The above image represents a batch size of 1, with below being both batch size = 16 and 32.

*Batch Size = 16*

Accuracy per Epoch             |  Accuracy vs. Training Time
:-------------------------:|:-------------------------:
<img width="500" alt="b16_full" src="https://user-images.githubusercontent.com/70443630/169362643-44129bd9-4ac8-454f-9d42-b9d8778c4e0b.PNG">  |  <img width="500" alt="b16_full_time" src="https://user-images.githubusercontent.com/70443630/169362749-d8c23dd8-c5d1-44ed-913e-c27693e052e2.PNG">


As we can see, the full dataset used with a batch size of 16 performs the best. However, training time was much shorter for the smaller dataset. Additionally, both the smaller datasets are still learning at the end of the 10th epoch. Each one would likely benefit from additional training cycles.

*Batch Size = 32*

Accuracy per Epoch             |  Accuracy vs. Training Time
:-------------------------:|:-------------------------:
 <img width="500" alt="b32_full" src="https://user-images.githubusercontent.com/70443630/169363377-bba5cb3b-f89e-4022-bec8-00c443f13c79.PNG">  |  <img width="500" alt="b32_full_time" src="https://user-images.githubusercontent.com/70443630/169363436-47b64069-8b06-4f41-88d9-27b655e5dc53.PNG">


Again, with a batch size of 32 we see on the left a much lower training time for the smaller sample-size datasets. What is different here is that the full dataset 


Below is a table of selected results, looking at both different sized datasets and different batch sizes. As we can see, the largest dataset tended to have higher test accuracy values. However, the total time these models took were substantially higher than the smaller datasets.

<p align = 'center'>
<img width="569" alt="results_batch_size" src="https://user-images.githubusercontent.com/70443630/169048621-f7ab94cd-a68a-460f-b652-a851ddcca345.PNG">
</p>

Lastly, as previously stated I wanted to see the impact that adding additional stop words to my standard list. Below is an image of training separate datasets on the same model, with the dataset with the standard stopword list performing much better.
<p align = 'center'>
<img width="400" alt="stopword_epoch" src="https://user-images.githubusercontent.com/70443630/169364954-aec94929-2f3e-49ba-83e0-0dc212c30647.PNG">
</p>
<p align = 'center'>
<img width="400" alt="stopwords table" src="https://user-images.githubusercontent.com/70443630/169590037-b3b880a3-694e-47a1-947c-fcfc5a54491c.PNG">
</p>

After testing several different and figuring out which combinations work the best, the following model was used:

<p align = 'center'>
<img width="400" alt="model_desc" src="https://user-images.githubusercontent.com/70443630/168905350-ccdfab4a-950e-449f-b2a2-61f64775a1c0.PNG">
</p>
  
The "best" model uses the downsampled dataset with 15,000 samples per emotion and a batch size of 32. It also only has one hidden layer with 256 hidden nodes. Lastly, since I am saving training time by using the smaller dataset, I trained the model for 25 epochs. Additionally, no extra stop words were added.  After testing, the highest accuracy achieved was **82.5%**, and I chose to save and use this model for predicting on the real-world articles.

## **Results**
---

- [Link to Article Testing](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Article_Testing.ipynb)

As mentioned above the articles being tested come from CNN and Fox News, obtained through scraping each site found [here](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Web_Scrape_CNN.ipynb) for CNN and [here](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Web_Scrape_Fox_News.ipynb) for Fox News. The updated web scrape is slightly skewed, with Fox News having only around 1/4 of the number of articles as CNN.

Each article is contained in a Pandas dataframe, and is preprocessed using the same cleaning techniques used on the huggingface dataset (tokenizing, stopword removal, lemmatization). After this, the text of each article is run through the saved model described above and a prediction of which emotion the article represents is sent back. 

### Article Predictions
---

<p align = 'center'>
<img width="400" alt="foxnews_predictions" src="https://user-images.githubusercontent.com/70443630/169464776-732b8213-8ab1-4264-84be-bf406a93a605.PNG">
</p>

In articles from Fox News, Joy leads the pack with the largest number of predicted articles. Fear is the second, with about half the number of predicted articles as joy, and is closely followed by anger articles. There were also no articles that were predicted as the emotion sadness.

<p align = 'center'>
<img width="400" alt="cnn_predictions" src="https://user-images.githubusercontent.com/70443630/169465161-d48ae4d1-72d1-4f97-a577-e9254593773f.PNG">
</p>

In similar findings to the articles from Fox News, but with a much larger sample size, the emotion of joy was the most popular prediction for articles from CNN. Fear and anger also followed the same pattern, with around half of the number of predictions as joy each.

the sentiment of articles from CNN were largely negative, with fear and surprise being the most popular emotion classified. 

<p align = 'center'>
<img width="400" alt="full_predictions" src="https://user-images.githubusercontent.com/70443630/169465522-01c37b8d-cf83-4213-b7a7-16804f270d68.PNG">
</p>

The full dataset again shows the same pattern as joy, anger, and fear are the 3 most popular emotions predicted. Love is not far behind as the 4th most predicted.

### Article Polarity
---

Each article tested was also assigned a polarity level to see where each one fell on a scale of -1 (negative) to 1 (positive). With joy being such as popular, it makes sense that the polarity of the overall dataset will skew towards the positive side (greater than 1). However, with anger and fear still being popular predictions we can see why there are still a few articles with a negative polarity. 

<p align = 'center'>
<img width="500" alt="polarity_hist" src="https://user-images.githubusercontent.com/70443630/169466927-e10c327c-d97b-490f-9333-0a24cdc2ab9c.PNG">
</p>

### Emotion Index
---
As mentioned above, I also wanted to test the level of fear predicted even when it wasn't the highest-predicted emotion. For this I created the fear index, which shows the distribution of the Softmax probabilities associated with fear for every single prediction. With 6 emotions a plurality is around 16%, so if most of the predictions fall around this point, this would mean strong predictions were not made  across the articles. 
It is easy to imagine that a similar model trained another time could give back very different predictions.


**Fear Index**             |  **Joy Index**
:-------------------------:|:-------------------------:
  <img width="400" alt="fear_index" src="https://user-images.githubusercontent.com/70443630/169468290-37be76d9-52f6-4762-ac2d-962a0904bfc8.PNG">  |  <img width="400" alt="joy_index" src="https://user-images.githubusercontent.com/70443630/169468320-1c0d2788-5326-4c88-bb9f-b71406079e9f.PNG">
  
 
 
 
 **Anger Index**             |  **Love Index**
:-------------------------:|:-------------------------:
<img width="400" alt="anger_index" src="https://user-images.githubusercontent.com/70443630/169469509-1bd0ee90-6161-4db7-bf69-e6ffc7b47b08.PNG">  |  <img width="400" alt="love_index" src="https://user-images.githubusercontent.com/70443630/169469557-32cce06b-e4fc-41e8-8dcb-0cf88ec2f1d2.PNG">

 
 
 
  **Sadness Index**             |  **Surprise Index**
:-------------------------:|:-------------------------:
 <img width="400" alt="sadness_index" src="https://user-images.githubusercontent.com/70443630/169469711-8520721f-4c6b-47ed-8645-5c1e1b06fa8e.PNG">  |  <img width="400" alt="surprise_index" src="https://user-images.githubusercontent.com/70443630/169468608-13c25c04-a36e-449e-a40a-e931019706ee.PNG">


As we see, each of the above emotions index's mean lies just around 0.2. However, all of the emotions still have samples that get closer to 0.5, with love being the only prediction to reach above 1/2. While this does mean that the model could improve in the long run, there are still strong predictions being made. It is also interesting to note that the histogram's of the negative emotions are very similar. 


## **Conclusions**
---

Let us finish off by looking back at the original research questions and how they fared against our samples from CNN and Fox News:
  
1. Is fear used more than other emotions within news articles? – **Among this sample of news articles from Fox News and CNN, joy was the highest-predicted emotion by far for both news sites. Fear and anger followed, having about half the amount of predictions as joy. Sadness was by far the least-predicted emotion.**
2. Is there a difference in the levels of fear used between different media outlets? – **From the collected articles, despite a different sample size, both Fox News and CNN had about the same distribution of emotion predictions. There is not a clear  difference in emotions used and each site follows the same pattern.**

## **Further Work**
---

1. *Continue Scraping Articles* - The greater the sample size of articles, the better trends we'll be able to see among articles and the main emotions they use. Additionally, bringing in more news sources (perhaps more sensational, "tabloid style" sources) would be interesting to see.
2. *Continue fine-tuning the model* - While the model I ended up using ended up working well on the huggingface dataset, there are likely better RNN architectures that would work better for transfer learning.

## **Resources**
---
 1. [The Fear-Industrial Complex](https://abcnews.go.com/2020/story?id=2898636&page=1)
 2. ["Bodybag Journalism"](https://www.chicagotribune.com/news/ct-xpm-1989-11-05-8901280504-story.html)
 3. [Multi-Class Sentiment Analysis on Twitter: Classification Performance and Challenges](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8681053)
 4. Raff, Edward: Inside Deep Learning [Link](https://www.manning.com/books/inside-deep-learning)
 5. [Fox News](foxnews.com)
 6. [CNN](cnn.com)
