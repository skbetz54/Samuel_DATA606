# The Fear-Industrial Complex
*An analysis on the amount of fear that plays into current US news media by Samuel Kolodrubetz*

## Topic Introduction

Anyone somewhat familiar with journalism and/or news media knows the quote "if it bleeds, it leads"<sup>1</sup>. For those who aren't, this essentially tells  that when violence, blood, or controversey are involved, it will likely get top billing. While this was originally stated in the late 19th century, the sentiment of the quote still rings true. Turn on any local TV news station and the top story will likely be about a murder, a scandal, or something similar. One thing is common under all of these leading stories, and lots of stories that cover news pages still: fear. Fear is one of the main factors that drives clicks/reads for every news outlet. But how much exactly do each of these outlets use this strategy of fear? Using sentiment analysis, I seek to look at this problem of fear in (online print) news media and its comprable impacts across various large outlets.

## Questions To Answer

1. Is there a quanitfiable measure of fear within certain news articles?
2. Is there a difference in the amount of fear used within a particular news topic across multiple news outlets?

## Data Source and Description

The main variable that will be measured is going to be the level of fear, which will be from now on be called the "Fear Index", that is present within the words, sentences, and paragraphs used in various articles of the same type (finance, politics, pop culture, etc.) across various news sources (AP News, CNN, Fox News). Each article counts as an indivual "entry" (think of each article as a person where the words and sentences being used are the x variables to explain the Fear Index y variable). 

The data is going to be scraped from each individual website (apnews.com, cnn.com, foxnews.com) with only the heading, subheading, and text being recorded (no pictures, captions, etc.). Ideally, depending on the success of the web scraper, I'd like to have at least 50 articles or more from each site from the same topic (I still haven't chosen which one; I think I'd like to go with either finance or politics because those are inherenty more negative).

## Intended Outcomes

Language is subjective; knowing what a writer/speaker intends to say when they write “shut up” takes context and understanding for the audience to know what they mean. Additionally, the proposed “Fear Index” that will be used to rank these articles/sources will be hard to 
With that being said, the number 1 outcome I intend to see is to be able to differentiate the Fear Index of two very different articles. For example, an article with the name “Cute Dog and Cute Cat Cuddle” would be expected to be very positive and not full of fear, while an article titled “World Ending in Less than a Year” would be regarded as a very negative article. The baseline outcome I expect is the first article to be very low on the Fear Index (0) with the latter being scored very high.  











*Title source: (ABC: "The Fear-Industrial Complex")[https://abcnews.go.com/2020/story?id=2898636&page=1])



## Sources
<sup>1</sup> - https://www.chicagotribune.com/news/ct-xpm-1989-11-05-8901280504-story.html - the first time I could find this quote being mentioned is in the 1898 Chicago Tribune article by Eleanor Randolph talking to another journalist about the state of print journalism all the way in the late 19th century. 
<sup>2</sup>
<sup>3</sup>
<sup>4</sup>
<sup>5</sup>
