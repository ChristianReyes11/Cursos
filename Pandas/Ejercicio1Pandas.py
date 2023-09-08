#1
#in the cell below, create a dataframe fruits that looks like this
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])
# Check your answer
q1.check()
fruits

#2 Create a dataframe fruit_sales that matches the diagram below:
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame([[35, 21], [41, 34]], columns=['Apples', 'Bananas'],
                index=['2017 Sales', '2018 Sales'])

# Check your answer
q2.check()
fruit_sales

#3.-
ingredients = pd.Series (['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'], name='Dinner')

# Check your answer
q3.check()
ingredients

#4.-
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

# Check your answer
q4.check()
reviews

#5.-
# Your code goes here
animals.to_csv('cows_and_goats.csv')

# Check your answer
q5.check()
