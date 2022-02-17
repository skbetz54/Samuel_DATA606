# The Fear-Industrial Complex<sup>1</sup>
*An analysis on the amount of fear that plays into current US news media by Samuel Kolodrubetz*

## Topic Introduction

Anyone somewhat familiar with journalism and/or news media is familiar with the quote "if it bleeds, it leads" <sup>2</sup>. For those who aren't, this essentially tells us that popular and leading stories often involve blood, death, controversial subjects, and other similar topics all revolving around one thing: **fear**. Fear is nowhere near a new concept, especially in journalism. The above quote was coined in the late 19<sup>th</sup> century, and that feeling still endures today. If you turn on any local TV station or read any online article the top story is likely going to be something that involves violence, a scandal, or a similar salacious event. 

Motivations for each individual news outlet differs as well. Nonprofits organizations do not seek to make profit from their reporting, and typically operate independently. For-profit organizations on the other hand operate with the intention to make a proift, and are often subject to control from the (public or private) financial sources. Additionally, each company and journalist will frame their stories differently. Despite these differences, fear is still a widely used tactic. But how much is it being used?


![Angry News Dude](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.ndtv.com%2Foffbeat%2Fmsnbc-anchor-lawrence-odonnells-angry-off-camera-meltdown-leaked-watch-viral-video-1753743&psig=AOvVaw250qKL--vxQgdoD0jXE9E-&ust=1645148120719000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCKC3xL7MhfYCFQAAAAAdAAAAABAD)

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

## Methodologies

This project will involve several steps that will utilize various techniques of a data science project:

1. Model Creation - Before quantifying the amount of fear in real-world articles, we need to train a model for sentiment analysis on our labeled emotion text data. To do this, I am going to be using Deep Learning. One of the most common models is a Recurrent Neural Network (RNN), which make processing sequences of data (sentences or articles) more effective. Because of my experience with it over other Deep Learning Python libraries, I will likely be using PyTorch and its associated packages. 
2. Data Scraping - As mentioned above, the individual data "entries" are going to be the individual articles pulled from the various news sites. This will be done using a Python library (either Scrapy, Beautifulsoup, or Selenium)
3. Data Cleaning - Preprocessing the incoming articles is incredibly important; text data needs to be put into a format that a machine can actually work with. Some important parts of this process are tokenization (breaking the article/sentence down into the simplest format for the machine to work with), stemming (finding the root of a given word, i.e from "eats" to "eat"), and removing unimportant "stop words" (words such as "the" or "and" which hold no value in NLP). This preprocessing will be done both for the model creation step and when analyzing the real-world articles.  

## Intended Outcomes

Language is subjective; knowing what a writer/speaker intends to say when they write “shut up” takes context and understanding for the audience to know what they mean. Because of this, I do not fully expect a completely successful classification of articles as fearful. However, a baseline result I expect is to have a well-performing model on a set of heavily curated articles that are chosen specifically for using fearful language throughout. For example, an article with the name “Cute Dog and Cute Cat Cuddle” would be expected to be very positive and not full of fear, while an article titled “World Ending in Less than a Year” would be regarded as a very negative article. The baseline outcome I expect is the first article to be very low on the Fear Index (0) with the latter being scored very high. 

For comparison across the various news outlets, I expect one main outcome: sites like AP News tend to sensationalize articles and headlines less than sites with political leanings like CNN and Fox News, so I would expect the overall fear level being used with AP News to be less than the other two. The other sites will, in my opinion be subjective to the articles that are actually scraped, but I would expect them to be relatively similar.


## Sources
<sup>1</sup> - https://abcnews.go.com/2020/story?id=2898636&page=1 - An ABC News article about how fear is a big factor in the stories that are followed, and the ones that ultimately find success. This article is the inspiration for this project's title.  
<sup>2</sup> - https://www.chicagotribune.com/news/ct-xpm-1989-11-05-8901280504-story.html - the first known appearance of the phrase "if it bleeds, it leads" comes from a New York Magazine article titled "Grins, Gore, and Videotape" by Eric Pooley. This Chicago Tribune article by Eleanor Randolph notes, this as I was unable to find the original article.

<sup>3</sup> - https://thehill.com/opinion/technology/556160-media-spread-fear-americans-listen
