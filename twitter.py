
import tweepy as tp
import os
import time
import chatbot
import json


credentials = json.load(open("credentials.json"))
# credentials to log in to Twitter API
consumer_key = 	credentials["Twitter"]["consumer_key"]
consumer_secret = credentials["Twitter"]["consumer_secret"]
access_token = credentials["Twitter"]["access_token"]
access_secret = credentials["Twitter"]["access_secret"]

print(consumer_key)
exit()

# Login to Twitter account API
auth = tp.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tp.API(auth)

api.update_status('Hey all')

