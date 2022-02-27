# The Fear-Industrial Complex<sup>1</sup>
*An analysis on the amount of fear that plays into current US news media by Samuel Kolodrubetz*

## Topic Introduction

Anyone somewhat familiar with journalism and/or news media is familiar with the quote "if it bleeds, it leads" <sup>2</sup>. For those who aren't, this essentially tells us that popular and leading stories often involve blood, death, controversial subjects, and other similar topics all revolving around one thing: **fear**. Fear is nowhere near a new concept, especially in journalism. The above quote was coined in the late 19<sup>th</sup> century, and that feeling still endures today. If you turn on any local TV station or read any online article the top story is likely going to be something that involves violence, a scandal, or a similar salacious event. 

Motivations for each individual news outlet differs as well. Nonprofits organizations do not seek to make profit from their reporting, and typically operate independently. For-profit organizations on the other hand operate with the intention to make a proift, and are often subject to control from the (public or private) financial sources. Additionally, each company and journalist will frame their stories differently. Despite these differences, fear is still a widely used tactic. But how much is it being used?

## Questions to Answer

This project looks at two questions regarding the use of fear within (online) news topics:


1. Is fear-based journalism still prevalent in the 2020s in a particular news topic (politics, finance, pop culture, etc.)? 
2. Is there a difference in the amount of fear used for different large media outlets and are nonprofit organizations less likely to use fearful language than opinionated media?

To investigate various news sources to perform a sentiment analysis to determine whether fear (among other emotions) is a common tactic in news outlets' stories. Since much of an article's information are captured in the first paragraph (with successive paragraphs adding additional information), the analysis will focus just on the opening paragraphs. 

## Data Source and Description

The first step is using Deep Learning to train a Recurrent Neural Network (RNN) on labeled text data from [huggingface.co](https://huggingface.co/datasets/emotion). The dataset contains 20,000 sentences and tweets which are classified as as one of 5 emotions: sadness, joy, love, anger and fear. It is split into training, validation, and test sets of 16,000/2,000/2,000.

**Note:** This dataset is similar to [one found on Kaggle](https://www.kaggle.com/pashupatigupta/emotion-detection-from-text), but that dataset focuses on tweets with over 13 emotions, but this is a separate dataset with only 5. [Huggingface dataset github](https://github.com/dair-ai/emotion_dataset)

Once the model(s) is trained and tested to a sufficient accuracy, I am then able to perform the analysis of emotions being used on the first paragraph of real-world articles. To do this, I first need to get the text of the first paragraph of articles from news  various outlets on the internet through a web scraping. In order to determine how fear and other emotions are used differently in different outlets, I will be considering three news outlets across the political spectrum: AP News, CNN, and Fox News. Additionally, since the style of reporting for different topics (politics, world news, finance, etc.) differs I am going to choose only world news articles to scrape to remain consistent. The unit of analysis, for both news articles and the labeled dataset, is the text of each individual article or sentence. 

For this kind of project where we are using text to analyze feelings of articles the more data gathered is always better to capture trends and find the outliers. Additionally, RNN arhitectures such as Long short-term memory (LTSM) perform much better when given more data, so I will start by taking as many articles as possible from each source (at least 100) with the same number being pulled from each source. 

Next, the variable of measure in this project is the emotion of the article/headline/sentence. While the main focus will be on the negative emotions of fear and anger, having a way to quantify other emotions could prove beneficial to the final outcome. With that, I can quantify the level of each emotion being shown within each article. For each article, the intended output will be a list of probabilities that the given input (article/sentence) belongs to each class (emotion). For example, a sentence reading "I am feeling sad" could have an output of (0.98, 0, 0, 0.01, 0.01), where the first value is the "saddness" class meaning the model predicts that sentence is showing sadness with a high probability. This is the baseline of measuring fear within each article, in what I will call the "Fear Index". The higher the probability the model gives to the "fear" class, the higher the Fear Index. Additionally, for measuring the relative use of fear across various news outlets, I will take into account each outlet's number of articles and the individual article's Fear Index. 

## **<ins>20<sup>th</sup>Exploratory Data Aanalysis<ins>**

- [Link to Happyface EDA notebook]()
- [Link to scraping notebook]() \*COMING SOON\*
  
This week's update is the beginning of my project; I am currently working on exploring the labeled dataset that contains tweets falling within 1 of 6 emotions. 
  
  

  
