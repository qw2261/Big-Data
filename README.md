- [COLUMBIA Big Data](#columbia-big-data)
  * [Homework 1](#-homework-1--https---githubcom-qw2261-big-data-tree-master-hw1-)
  * [Homework 2](#-homework2--https---githubcom-qw2261-big-data-tree-master-hw2-)

# COLUMBIA Big Data 

The course projects of Columbia Big Data Analytics

## [Homework 1](https://github.com/qw2261/Big-Data/tree/master/HW1)



1. Implement and run in Spark
2. Process data with Spark Dataframe, and perform graph analysis 

The goals of this assignment are to 

(1) understand how to implement K-means clustering algorithm in Spark by utilizing *transformations and actions*, 

(2) understand the impact of using different distance measurements and initialization strategies in clustering, 

(3) learn how to use the built-in Spark MLlib library to conduct supervised and unsupervised learning, 

(4) have experience of processing data with *ML Pipeline* and *Dataframe*. 

In the first question, you will conduct **document clustering**. The dataset we’ll be using is a set of vectorized text documents. In today’s world, you can see applications of document clustering almost everywhere. For example, Flipboard uses LDA topic modelling, approximate nearest neighbor search, and clustering to realize their “similar stories / read more” recommendation feature. You can learn more by reading this blog post. To conduct document clustering, you will implement the classic iterative K-means clustering in Spark with different distance functions, and compare with the one implemented in Spark MLlib. 

![](https://github.com/qw2261/Big-Data/blob/master/Pics/hw1_2.png)

In the second question, you will load data into Spark Dataframe and perform **binary classification** with Spark MLlib. We will use logistic regression model as our classifier, which is one of the foundational and widely used tools for making classifications. For example, Facebook uses logistic regression as one of the components in its online advertising system. You can read more in a publication [here](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf). 



## [Homework 2](https://github.com/qw2261/Big-Data/tree/master/HW2)



Write a Spark program that implements a simple “People You Might Know” social network friendship recommendation algorithm. The key idea is that if two people have a lot of mutual friends, then the system should recommend that they connect with each other. 

*Input:* 

The input file contains the adjacency list and has multiple lines in the following format: <User> <TAB> <Friends>. Here, <User> is a unique integer ID corresponding to a unique user and <Friends> is a comma separated list of unique IDs corresponding to the friends of the user with the unique ID <User>. Note that the friendships are mutual (i.e., edges are undirected): if A is friend with B then B is also friend with A. The data provided is consistent with that rule as there is an explicit entry for each side of each edge. 

*Algorithm:* 

Let us use a simple algorithm such that, for each user U, the algorithm recommends N = 10 users who are not already friends with U, but have the most number of mutual friends in common with U. 

*Output:* 

The output should contain one line per user in the following format: <User><Recommendations> Where <User> is a unique ID corresponding to a user and <Recommendations> is a list of unique IDs corresponding to the algorithm’s recommendation of people that <User> might know, ordered in decreasing number of mutual friends. If a user has less than 10 second-degree friends, output all of the, in decreasing order of the number of mutual friends. If a user has no friends, providing an empty list of recommendations. If there are recommended users with the same number of mutual friends, then output those user IDs in numerically ascending order. 



Run Connected Components and PageRank with GraphFrames. You can refer to the GraphFrames documentation: https://graphframes.github.io/graphframes/docs/_site/index.html.



PageRank measures the importance of each vertex in a graph, assuming an edge from u to v represents an endorsement of v’s importance by u. For example, if a Twitter user is followed by many others, the user will be ranked highly. 
