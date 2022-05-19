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
- [Link to Article Testing](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Article_Testing.ipynb)




## Topic Introduction
---

Anyone familiar to American journalism and/or news media is familiar with the quote "if it bleeds, it leads" <sup>2</sup>. For those who aren't, this essentially tells us that popular and leading stories often involve blood, death, controversial subjects, and other similar topics all revolving around one thing: **fear**. Fear is nowhere near a new concept, especially in journalism. The above quote was coined in the late 19<sup>th</sup> century, and that feeling still endures today. If you turn on any local TV station or read any online article the top story is likely going to be something that involves violence, a scandal, or a similar salacious event. 

Motivations for each individual news outlet differs as well. Nonprofits organizations do not seek to make profit from their reporting, and typically operate independently. For-profit organizations on the other hand operate with the intention to make a proift, and are often subject to control from the (public or private) financial sources. Additionally, each company and journalist will frame their stories differently. Despite these differences, fear is still a widely used tactic. But how much is it being used?

## Questions to Answer
---

This project looks at two questions regarding the use of fear within (online) news topics:


1. Is fear-based journalism still prevalent in the 2020s in present-day news media? 
2. Is there a difference in the amount of fear used among different large media outlets?


## Data Source and Description
---

The first step is using deep learning to train a Recurrent Neural Network (RNN) on labeled text data from [huggingface.co](https://huggingface.co/datasets/emotion). The dataset contains 416,806 sentences and tweets which are classified as as one of 6 emotions: sadness, joy, love, surprise, anger, and fear.

**Note:** This dataset is similar to [one found on Kaggle](https://www.kaggle.com/pashupatigupta/emotion-detection-from-text), but that dataset focuses on tweets with over 13 emotions, but this is a separate dataset with only 5. [Huggingface dataset github](https://github.com/dair-ai/emotion_dataset)

Once the model(s) is trained and tested to a sufficient accuracy, I am then able to perform the analysis of emotions being used in real-world articles. To do this, I'll first create a Pandas dataframe with the articles' date, headline, and text (obtained using the Python library Newspaper3k). In order to determine how fear and other emotions are used differently in different outlets, I will be considering two news outlets across the political spectrum: CNN and Fox News. Additonally, after some investigation the sensational languages being used occurr much more frequently within opinion articles. For this reason, articles scraped for this project are opinion pieces hosted on these sites.

The variable of measure in this project is the emotion of the article/headline/sentence. The main focus is going to be building a model that will be able to accurately classify an input string (tweet or article) as any of the 6 emotions based on the labeled huggingface dataset. This classification is done by using a softmax output for each input string, which assigns a list of probabilities to each emotion based on how likely it is to be the correct answer. The highest probability is then selected and used as the output's prediction. For example, a sentence reading "I am feeling sad" could have an output of (0.98, 0, 0, 0.01, 0.01), where the first value is the "saddness" class, which would return "Sadness" as the prediction. While an intended unit of measure was to add a "fear index", which would be the associated probability of the fear class (even if it wasn't the highest likelihood) I ended up focusing on the final output of the softmax function.


## **Exploratory Data Aanalysis**
---

- [Link to Happyface EDA notebook](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/EDA_Hugging_Face.ipynb)

The first part of my project entails creating a machine learning model (specifically a Recurrent Neural Network) using the labeled Huggingface emotion data, but before I dive into the modeling, I need to understand the dataset a little better. This entails both understanding the entire dataset, and also the difference between classes within the dataset. For example, in the image below we see that sadness and joy have a much higher number of samples compared to other emotions. This tells us that in the model creation, we will have to deal with this class imbalance by downsampling the high-frequency emotions.

<img width="500" alt="frequency chart" src="https://user-images.githubusercontent.com/70443630/155910809-a578cbf8-e554-4b90-a0e8-38f26061dae0.PNG">

Another point of focus in this Exploratory Analysis is the word frequency for each emotion. When looking at the top 10 words used for fear and surprise, there is not much we can gather; there are only a couple words that are associated with the emotion (fear --> afraid, surprised --> shock). While this does not tell us much about unique words seen in each class, it can help us dictate what words NOT to include in the model building. For example, the word "really" and "like" appear often in every single class. Adding these types of words prevelant throughout the dataset for each class to the list of stopwords is something that I wanted to end up testing its impact.

<img width="500" alt="freq_fear" src="https://user-images.githubusercontent.com/70443630/155911189-b88e9b82-a42c-4eae-9f42-82157a3ee4e6.PNG"><img width="500" alt="freq_surprise" src="https://user-images.githubusercontent.com/70443630/155911193-9f31ffb9-d358-4b20-9cce-1c9c9632d849.PNG">

However, if we expand our search and include more of the top words being used for each emotion, we start to see a better picture of the unique vocabularies being used with the model's recognizing each emotion. To do this, I created wordclouds for each emotion and the frequency of words being used, with the top 100 words being shown.

First for fear:

<img width="600" alt="wordcloud_fear" src="https://user-images.githubusercontent.com/70443630/155911349-ca550de0-56d3-4340-b5a9-42a512b692ad.PNG">

And for surprised:

<img width="600" alt="wordcloud_surprised" src="https://user-images.githubusercontent.com/70443630/155911377-2932d035-b9d2-44ee-a088-b1cb031e2bab.PNG">

Both of these wordclouds show great examples of unique words that are present within these emotions. In the wordcloud for fear, words such as "nervious", "afraid", and "anxious" are likely to not be very prevalent within other emotions (at least for the more positive emotions). Similarly, surprise shows words that can only be asssociated with surprise such as "shock", "overwhelm", or "stun".

Lastly I took a look at the polarity of each individual emotion's dataset using Textblob's built in sentiment analysis library. This function assigns a polarity value between -1 and 1, where -1 is a more negative sentiment and 1 is the most positive sentiment. As we see from the below group of histograms, the negative emotions (top row) are all centered around 0 (neutral) with an obvious skew to the left towards a negative sentiment. On the other hand, the positive emotions (bottom row) are again centered around 0 but is skewed towards a more positive sentiment.

<img width="700" alt="hist_polarity" src="https://user-images.githubusercontent.com/70443630/169345576-e40743e1-048f-47db-ba18-e5a465c1d9f7.PNG">

  
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

<img width="400" alt="frequency chart - std" src="https://user-images.githubusercontent.com/70443630/168902050-67baa6a8-5447-4386-a58a-219f3e18d619.PNG">

50K Downsampled Dataset

<img width="400" alt="frequency chart - 50k" src="https://user-images.githubusercontent.com/70443630/168902097-5983101a-0346-46d7-9ef8-32c8f7fedf72.PNG">

15K Sample Datset

<img width="400" alt="frequency chart - 15k" src="https://user-images.githubusercontent.com/70443630/168902139-3d4ee574-e83c-4626-9f55-f94586f7dce0.PNG">



As we can see below, while the smaller datasets increased their accuracy slightly slower each epoch, the time taken for the full dataset was much longer. For that reason, the final "best" model was trained and tested on the dataset with 15,000 samples for each class. 

<img width="500" alt="b1_full_time" src="https://user-images.githubusercontent.com/70443630/168902624-b32fd28b-a2e2-4a8e-b02d-24d4f1a9c0fb.PNG">

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


<img width="569" alt="results_batch_size" src="https://user-images.githubusercontent.com/70443630/169048621-f7ab94cd-a68a-460f-b652-a851ddcca345.PNG">


Lastly, as previously stated I wanted to see the impact that adding additional stop words to my standard list. Below is an image of training separate datasets on the same model, with the dataset with the standard stopword list performing much better.

<img width="400" alt="stopword_epoch" src="https://user-images.githubusercontent.com/70443630/169364954-aec94929-2f3e-49ba-83e0-0dc212c30647.PNG">


After testing several different and figuring out which combinations work the best, the following model was used:

<img width="400" alt="model_desc" src="https://user-images.githubusercontent.com/70443630/168905350-ccdfab4a-950e-449f-b2a2-61f64775a1c0.PNG">

The "best" model uses the downsampled dataset with 15,000 samples per emotion and a batch size of 32. It also only has one hidden layer with 256 hidden nodes. Lastly, since I am saving training time by using the smaller dataset, I trained the model for 25 epochs. Additionally, no extra stop words were added.  After testing, the highest accuracy achieved was **82.5%**, and I chose to save and use this model for predicting on the real-world articles.

**Results**
---

- [Link to Article Testing](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Article_Testing.ipynb)

As mentioned above the articles being tested come from CNN and Fox News, obtained through scraping each site found [here](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Web_Scrape_CNN.ipynb) for CNN and [here](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Web_Scrape_Fox_News.ipynb) for Fox News. 50 articles were obtained from each site. The breakdown of the scraped articles are as follows:

<img width="500" alt="table_scrape" src="https://user-images.githubusercontent.com/70443630/168900045-86f8728a-fa00-49af-94b1-906be5a6954b.PNG">

Each article is contained in a Pandas dataframe, and is preprocessed using the same cleaning techniques used on the huggingface dataset (tokenizing, stopword removal, lemmatization). After this, the text of each article is run through the saved model described above and a prediction of which emotion the article represents is sent back. 


<img width="500" alt="predictions_fn" src="https://user-images.githubusercontent.com/70443630/168871437-f98fd429-018c-44ed-afcf-b38128fd7966.PNG">

In Articles from Fox News we see that fear is indeed a popular emotion used in the selected news articles, followed closely by surprise, anger, and sadness. Another notable finding is that aside from surprise, positive emotions (joy and love) were not predicted to be many of the articles' sentiment.


<img width="500" alt="predictions_cnn" src="https://user-images.githubusercontent.com/70443630/168871431-0f0b798f-c6a9-4f0b-9a57-85c3b3e7e108.PNG">

In similar findings to the articles from Fox News, the sentiment of articles from CNN were largely negative, with fear and surprise being the most popular emotion classified. 

<img width="500" alt="predictions_full" src="https://user-images.githubusercontent.com/70443630/168871457-785acb27-f657-48d6-a4d8-1d04703abe1e.PNG">

The full dataset again shows that fear and surprise are the most widely used emotions.

<img width="500" alt="polarity_hist" src="https://user-images.githubusercontent.com/70443630/168871418-b719e60a-60d1-40a6-8b98-79ead023feed.PNG">

Each article tested was also assigned a polarity level to see where each one fell on a scale of -1 (negative) to 1 (positive). Similar to the histogram of the huggingface data, the histogram for each article's polarity skews to the negative side, but again it stays much closer to 0 (neutral) than previously expected.

## **Conclusions**
---

Let us finish off by looking back at the original research questions and how they fared against our samples from CNN and Fox News.
  
1. Is fear used more than other emotions within news articles? – **Fear and surprise are the most used emotions within real-world articles. More positive emotions (love and joy) are used much less.**
2. Is there a difference in the levels of fear used between different media outlets? – **From the collected articles, there is no noticeable difference in emotions used. Each site follows the same pattern.**

