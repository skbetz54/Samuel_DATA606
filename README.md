# The Fear-Industrial Complex<sup>1</sup>
*An analysis on the amount of fear that plays into current US news media by Samuel Kolodrubetz*

## Links

- [Link to full proposal](https://github.com/skbetz54/Samuel_DATA606/blob/main/Approved_Proposal.md)
- [Link to proposal video presentation](https://www.youtube.com/watch?v=F_P-RPXwhEE)
- [Link to proposal presentation](https://github.com/skbetz54/Samuel_DATA606/blob/main/Samuel%20Kolodrubetz%20-%20Fear%20Industrial%20Complex.pptx)
- [Link to EDA](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/EDA_Hugging_Face.ipynb)
- [Link to Machine Learning](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Machine_Learning_Implementation.ipynb)
- [Link to Web Scrape - CNN](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/CNN.ipynb)
- [Link to Web Scrape - Fox News](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Web_Scrape_Fox_News.ipynb)
- [Link to Article Testing](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/Article_Testing.ipynb)




## Topic Introduction

Anyone familiar to American journalism and/or news media is familiar with the quote "if it bleeds, it leads" <sup>2</sup>. For those who aren't, this essentially tells us that popular and leading stories often involve blood, death, controversial subjects, and other similar topics all revolving around one thing: **fear**. Fear is nowhere near a new concept, especially in journalism. The above quote was coined in the late 19<sup>th</sup> century, and that feeling still endures today. If you turn on any local TV station or read any online article the top story is likely going to be something that involves violence, a scandal, or a similar salacious event. 

Motivations for each individual news outlet differs as well. Nonprofits organizations do not seek to make profit from their reporting, and typically operate independently. For-profit organizations on the other hand operate with the intention to make a proift, and are often subject to control from the (public or private) financial sources. Additionally, each company and journalist will frame their stories differently. Despite these differences, fear is still a widely used tactic. But how much is it being used?

## Questions to Answer

This project looks at two questions regarding the use of fear within (online) news topics:


1. Is fear-based journalism still prevalent in the 2020s in present-day news media? 
2. Is there a difference in the amount of fear used among different large media outlets?

To investigate various news sources to perform a sentiment analysis to determine whether fear (among other emotions) is a common tactic in news outlets' stories.
Since much of an article's information are captured in the first paragraph (with successive paragraphs adding additional information), the analysis will focus just on the opening paragraphs. 

## Data Source and Description

The first step is using Deep Learning to train a Recurrent Neural Network (RNN) on labeled text data from [huggingface.co](https://huggingface.co/datasets/emotion). The dataset contains 20,000 sentences and tweets which are classified as as one of 6 emotions: sadness, joy, love, surprise, anger, and fear. It is split into training, validation, and test sets of 16,000/2,000/2,000.

**Note:** This dataset is similar to [one found on Kaggle](https://www.kaggle.com/pashupatigupta/emotion-detection-from-text), but that dataset focuses on tweets with over 13 emotions, but this is a separate dataset with only 5. [Huggingface dataset github](https://github.com/dair-ai/emotion_dataset)

Once the model(s) is trained and tested to a sufficient accuracy, I am then able to perform the analysis of emotions being used in real-world articles. To do this, I'll first create a Pandas dataframe with the articles' date, title, and text (obtained using the Python library Newspaper3k). In order to determine how fear and other emotions are used differently in different outlets, I will be considering two news outlets across the political spectrum: CNN and Fox News. Additonally, after some investigation the sensational languages being used occurr much more frequently within opinion articles. For this reason, articles scraped for this project are opinion pieces hosted on these sites.

The variable of measure in this project is the emotion of the article/headline/sentence. While the main focus will be on the negative emotions of fear and anger, having a way to quantify other emotions could prove beneficial to the final outcome. With that, I can quantify the level of each emotion being shown within each article. For each article, the intended output will be a list of probabilities that the given input (article/sentence) belongs to each class (emotion). For example, a sentence reading "I am feeling sad" could have an output of (0.98, 0, 0, 0.01, 0.01), where the first value is the "saddness" class meaning the model predicts that sentence is showing sadness with a high probability. This is the baseline of measuring fear within each article, in what I will call the "Fear Index". The higher the probability the model gives to the "fear" class, the higher the Fear Index. Additionally, for measuring the relative use of fear across various news outlets, I will take into account each outlet's number of articles and the individual article's Fear Index. 

## **Exploratory Data Aanalysis**

- [Link to Happyface EDA notebook](https://github.com/skbetz54/Samuel_DATA606/blob/main/Notebooks/1_1_EDA_Hugging_Face.ipynb)

The first part of my project entails creating a machine learning model (specifically a Recurrent Neural Network) using the labeled Huggingface emotion data, but before I dive into the modeling, I need to understand the dataset a little better. This entails both understanding the entire dataset, and also the difference between classes within the dataset. For example, in the image below we see that sadness and joy have a much higher number of samples compared to other emotions. This tells us that in the model creation, we will have to deal with this class imbalance (likely by downsampling the high-frequency emotions).

<img width="500" alt="frequency chart" src="https://user-images.githubusercontent.com/70443630/155910809-a578cbf8-e554-4b90-a0e8-38f26061dae0.PNG">

Another point of focus in this Exploratory Analysis is the word frequency for each emotion. When looking at the top 10 words used for fear and surprise, there is not much we can gather; there are only a couple words that are associated with the emotion (fear --> afraid, surprised --> shock). While this does not tell us much about unique words seen in each class, it can help us dictate what words NOT to include in the model building. For example, the word "really" and "like" appear often in every single class. Adding these to the list of stopwords (common words that would not add any context or improve the model) is a possibility when reaching the model creation stage.

<img width="500" alt="freq_fear" src="https://user-images.githubusercontent.com/70443630/155911189-b88e9b82-a42c-4eae-9f42-82157a3ee4e6.PNG"><img width="500" alt="freq_surprise" src="https://user-images.githubusercontent.com/70443630/155911193-9f31ffb9-d358-4b20-9cce-1c9c9632d849.PNG">

However, if we expand our search and include more of the top words being used for each emotion, we start to see a better picture of the unique lexicons being used within each emotion. To do this, I created wordclouds for each emotion and the frequency of words being used, with the top 100 words being shown.

First for fear:

<img width="600" alt="wordcloud_fear" src="https://user-images.githubusercontent.com/70443630/155911349-ca550de0-56d3-4340-b5a9-42a512b692ad.PNG">

And for surprised:

<img width="600" alt="wordcloud_surprised" src="https://user-images.githubusercontent.com/70443630/155911377-2932d035-b9d2-44ee-a088-b1cb031e2bab.PNG">

  
## **Machine Learning Implementation and Results**

Now that the data is cleaned and ready to be used within a deep learning model, I can now build a model to capture how much fear is being used within current news media.

The real-world articles used come from CNN and Fox News, with 50 articles from each site.


<img width="209" alt="predictions_cnn" src="https://user-images.githubusercontent.com/70443630/168871431-0f0b798f-c6a9-4f0b-9a57-85c3b3e7e108.PNG">

Within CNN, we see that fear and surprise are both very common predictions. 

<img width="141" alt="predictions_fn" src="https://user-images.githubusercontent.com/70443630/168871437-f98fd429-018c-44ed-afcf-b38128fd7966.PNG">

In Articles from Fox News we see a very similar distribution of emotions being predicted upon. Fear and surprise are among the most common emotions illicited, with joy and love being the least popular.

<img width="211" alt="predictions_full" src="https://user-images.githubusercontent.com/70443630/168871457-785acb27-f657-48d6-a4d8-1d04703abe1e.PNG">

The full dataset again shows that fear and surprise are the most widely used emotions.

<img width="214" alt="polarity_hist" src="https://user-images.githubusercontent.com/70443630/168871418-b719e60a-60d1-40a6-8b98-79ead023feed.PNG">

Each article tested was also assigned a polarity level to see where each one fell on a scale of -1 (negative) to 1 (positive). Similar to the histogram of the huggingface data, the histogram for each article's polarity skews to the negative side, but again it stays much closer to 0 (neutral) than previously expected.

## **Conclusions**
  
1. Is fear used more than other emotions within news articles? – **Fear and surprise are the most used emotions within real-world articles. More positive emotions (love and joy) are used much less.**
2. Is there a difference in the levels of fear used between different media outlets? – **From the collected articles, there is no noticeable difference in emotions used. Each site follows the same pattern.**

