import pandas as pd

wine_reviews = pd.read_csv("./winemag-data-130k-v2.csv")

# selecto top 1 
print('\n sub-dataset selection wine is italian or not \n select country from winesmag\n')
wine_reviews_italy = wine_reviews.loc [0, 'country']
print(wine_reviews_italy)
#select taster_name , taster_tiwiter
wine_reviews_italy = wine_reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points' ]]
print(wine_reviews_italy)
#select title as index
wine_reviews3 = wine_reviews.set_index('title')
print(wine_reviews3)
# select is Italy (country) from wines -- sql 
wines_italy = wine_reviews.country == 'Italy'
print(wines_italy)
# select from wine here country = 'Italy'
wines_all_italy = wine_reviews.loc[wine_reviews.country == 'Italy']
print(wines_all_italy)
# select from winea where country= 'italy'd points >=90
wines_all_points = wine_reviews.loc[(wine_reviews.country == 'Italy') & (wine_reviews.points >= 90)], ['country' , 'points']
print(wines_all_points)
# select from winea where country= 'italy'd points >=90
#wines_all_points = wine_reviews.loc[(wine_reviews.country == 'Italy') || (wine_reviews.points >= 90)], ['country' , 'points']
#print(wines_all_points)
