import pandas as pd
import numpy as np
import math
from random import *


minNumberOfRatingsForEachBook = 100        # Leaves us with 6247 book items
minNumberOfRatingsForEachUser = 10         # Leaves us with 16206 users


allRatings = pd.read_csv("ratings.csv")
books = pd.read_csv("books.csv")
allRatings = allRatings.drop_duplicates(subset=['book_id', 'user_id'])


numberOfRatingsOfEachBook = allRatings["book_id"].value_counts()
eligibleBookItems = numberOfRatingsOfEachBook[numberOfRatingsOfEachBook >= minNumberOfRatingsForEachBook].index    # Keep only those books that have been rated by 100 or more users
filteringCondition = allRatings["book_id"].apply(lambda x: x in eligibleBookItems)
allRatings = allRatings[filteringCondition]


numberOfRatingsOfEachUser = allRatings["user_id"].value_counts()
eligibleUsers = numberOfRatingsOfEachUser[numberOfRatingsOfEachUser >= minNumberOfRatingsForEachUser].index    # Keep only those books that have been rated by 100 or more users
filteringCondition = allRatings["user_id"].apply(lambda x: x in eligibleUsers)
allRatings = allRatings[filteringCondition]

contentMatrix = allRatings.pivot(index="user_id", columns="book_id", values="rating")

userIDs = allRatings['user_id']
userIDs.drop_duplicates(inplace=True)
userIDs = userIDs.sort_values()
userIDs = userIDs.reset_index(drop=True)



def getNullAndNonNullColumnsLists(user, dataFrame):   # Get list of non null and null columns of the 'user' . Columns represents books. This function will actually give way for a user which books have ratings and which do not.
    nonNullIndexes = []
    nullIndexes = []
    bookIDs = contentMatrix.columns
    
    for book in bookIDs:
        if pd.isnull(dataFrame.loc[user, book]) == False:
            nonNullIndexes.append(book)
        else:
            nullIndexes.append(book)
    return (nonNullIndexes, nullIndexes)
    
    
    
def recommendBooks(targetUser):
    (ratedItems, unRatedItems) = getNullAndNonNullColumnsLists(targetUser, contentMatrix)
    ratingsOfUnratedItems = []
    
    count = 0
    iteration = 0
    
    for unRatedItem in unRatedItems:
        iteration += 1
        similarityDataOfUnRatedItem = []
        ratedItemsSetOfRandomUser = set(ratedItems)
        df = contentMatrix[ratedItems]
        df = df.dropna(axis=0, how="all")                   # Removing users that haven't rated at least one item same as the randomUser
        probableSimilarUsers = list(df.index.values)        # Users that have rated at least one item same as the randomUser
        usersWhoHaveRatedThisItem = list(contentMatrix[[unRatedItem]].dropna().index)
        filteredProbableUsers = []
        chosenUsers = []
        probableSimilarUsers = list(set(probableSimilarUsers) & set(usersWhoHaveRatedThisItem))
       
       
        count2 = 0
        
        for currentUser in probableSimilarUsers:
            count2 += 1
            (ratedItemsOfThisUser, unRatedItemsOfThisUser) = getNullAndNonNullColumnsLists(currentUser, contentMatrix)
            ratedItemsSetOfThisUser = set(ratedItemsOfThisUser)
            commonRatedItems = ratedItemsSetOfRandomUser & ratedItemsSetOfThisUser
            commonRatedItems = pd.Series(sorted(list(commonRatedItems)))
            
            
            # "targetUser" is the user for whose non rated items we are going to calculate ratings
            # "currentUser" is the user with whom we will compare our targetUser and calculate similarity

            currentUserRatings = [x for x in contentMatrix.loc[currentUser, list(commonRatedItems)]]         # Ratings of common items by this user
            targetUserRatings = [x for x in contentMatrix.loc[targetUser, list(commonRatedItems)]]    # Ratings of common items by targetUser
          

            currentUserAverage = sum(currentUserRatings) / len(currentUserRatings)      # Average rating of the currentUser
            targetUserAverage = sum(targetUserRatings) / len(targetUserRatings)         # Average rating of the targetUser

            tempList1 = [ (ra - targetUserAverage) * (rb - currentUserAverage)  for (ra,rb) in zip(targetUserRatings, currentUserRatings) ]
            tempList2 = [ (ra - targetUserAverage) * (ra - targetUserAverage) for ra in targetUserRatings ]   
            tempList3 = [ (rb - currentUserAverage) * (rb - currentUserAverage) for rb in currentUserRatings ]
            tempList4 = [ (rb - currentUserAverage) for rb in currentUserRatings ]
            
            
            print("For User {} Selected items so far: {}     Unrated Item {} of {}   Checked similarity on users: {}/{}".format(targetUser, count, iteration, len(unRatedItems), count2, len(probableSimilarUsers)))
       
            if ((sum(tempList1)) == 0):
                similarity = 0
            else:
                similarity = round((sum(tempList1)) / (math.sqrt(sum(tempList2)) * math.sqrt(sum(tempList3))), 2)
            if similarity > 0:
                chosenUsers.append(currentUser)
                similarityDataOfUnRatedItem.append((similarity,(contentMatrix.loc[currentUser, unRatedItem] - currentUserAverage)))
                    
        
        if(len(similarityDataOfUnRatedItem) > 0):       # This means some similar users exist from which we can calculate rating for this unrated item for our current targetUser
            similarityDataOfUnRatedItem = sorted(similarityDataOfUnRatedItem, reverse=True)
            similarityDataOfUnRatedItem = similarityDataOfUnRatedItem[:50]                     # Selecting 50 most similar users' similarity info
            predictedRatingOfCurrentUnratedItem = targetUserAverage + (sum([x[0] * x[1] for x in similarityDataOfUnRatedItem])/sum([x[0] for x in similarityDataOfUnRatedItem]))
            ratingsOfUnratedItems.append((predictedRatingOfCurrentUnratedItem, unRatedItem))
            count += 1
            print("For User {} Selected items so far: {}     Unrated Item {} of {}   Checked similarity on users: {}/{}".format(targetUser, count, iteration, len(unRatedItems), count2, len(probableSimilarUsers)))
            
        else:           # This else block means for this unrated item of our current target user, no similar user has been found. Hence rating for this unrated item can not be calculated
            print("For User {} Selected items so far: {}     Unrated Item {} of {}   Checked similarity on users: {}/{}".format(targetUser, count, iteration, len(unRatedItems), count2, len(probableSimilarUsers)))
            continue

    print()
    print()
    print()
    if( len(ratingsOfUnratedItems) > 0):
        ratingsOfUnratedItems = sorted(ratingsOfUnratedItems, reverse=True)[:5]
        print("Recommendations for {}:".format(targetUser))
        count = 1
        for newRatedItem in ratingsOfUnratedItems:
            newRating, book = newRatedItem
            if(newRating < 1):
                newRating = 1.0           # The lowest value of rating should be 1
            elif(newRating > 5):
                newRating = 5.0           # The highest value of rating should be 5
            else:
                newRating = float(int(newRating))    # Getting rid of anything after decimal
            contentMatrix.loc[[targetUser], [book]] = newRating     # Updating the contentMatrix with the new rating
            print()
            print("{}. {}     Predicted Rating: {}".format(count, books[books["id"] == book]["title"][book-1], newRating))
            print()
            count += 1
    else:
        print("Content matrix is not rich enough for this user to generate ratings for any of his unrated items. Try this user again later after some iterations of recommendations.")
        


while True:
    print()
    print()
    print("Choose from one of the following options (Enter an option number):")
    print("1. Randomly select a user and show recommended books for him")
    print("2. Enter a user id and show recommended books for him")
    print("3. Quit")
    selectedOption = input()
    print(selectedOption)
    if int(selectedOption) == 1:
        targetUser = userIDs[randrange(0, len(userIDs))]
        recommendBooks(targetUser)
    elif int(selectedOption) == 2:
        print(userIDs)
        print()
        recommendBooks(int(input("Choose and enter a user id from above to get recommendation for him: ")))
    elif int(selectedOption) == 3:
        break
    else:
        print("Invalid input given")
